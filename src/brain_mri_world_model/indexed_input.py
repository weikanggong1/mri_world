"""Compile complete encoding/corpus metadata once into an immutable, lazy SQLite input.

Only producer JSON/JSONL/COMPLETE files are opened. MRI and latent tensors are never
opened, statted or hashed by this module. One bounded child uses mature validation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

VERSION='brain_mri_indexed_input_runtime_v1'
MARKER='INDEX_COMPLETE'
DB_NAME='records.sqlite3'
SPLITS=('train','val','test')

def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def _fingerprint(path):
    from .integrity import fingerprint
    return list(fingerprint(path))

def _json_file(path):
    from .corpus import _json_file as mature_json
    return mature_json(Path(path))

def _write(path,value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())

def _schema(db):
    db.executescript('''PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA user_version=1;
      CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE subjects(dataset TEXT,subject_id TEXT,split TEXT NOT NULL,PRIMARY KEY(dataset,subject_id));
      CREATE TABLE images(image_index INTEGER PRIMARY KEY,dataset TEXT NOT NULL,image_id TEXT NOT NULL,
        subject_id TEXT NOT NULL,modality TEXT NOT NULL,split TEXT NOT NULL,representation TEXT NOT NULL,
        source_identity TEXT UNIQUE NOT NULL,voxel_sha256 TEXT,first_record_index INTEGER,axes_mask INTEGER NOT NULL DEFAULT 0,
        record0 INTEGER,record1 INTEGER,record2 INTEGER,splitrecord0 INTEGER,splitrecord1 INTEGER,splitrecord2 INTEGER,
        shape0 TEXT,shape1 TEXT,shape2 TEXT,UNIQUE(dataset,image_id));
      CREATE TABLE records(record_index INTEGER PRIMARY KEY,split TEXT NOT NULL,split_index INTEGER NOT NULL,
        image_index INTEGER NOT NULL,axis INTEGER NOT NULL,shape TEXT NOT NULL,latent_path TEXT UNIQUE NOT NULL,
        latent_sha256 TEXT UNIQUE NOT NULL,record_json TEXT NOT NULL,
        UNIQUE(split,split_index),UNIQUE(image_index,axis));
      CREATE TABLE voxel_splits(representation TEXT,voxel_sha256 TEXT,split TEXT NOT NULL,
        PRIMARY KEY(representation,voxel_sha256));
      CREATE INDEX record_split_order ON records(split,record_index);
      CREATE INDEX image_split_order ON images(split,first_record_index);
      CREATE INDEX image_runtime_subject ON images(dataset,subject_id);
      CREATE INDEX image_global_order ON images(first_record_index);''')

def eligible_logical_image_counts(db,raw_images_by_split,held_images):
    """Count each held canonical image once, independent of its XYZ/native axis rows."""
    held_by_split=Counter()
    for image_index in sorted(held_images):
        row=db.execute('SELECT split FROM images WHERE image_index=?',(image_index,)).fetchone()
        if row is None or row[0] not in SPLITS:raise ValueError('Held image is absent from its canonical image grouping')
        held_by_split[row[0]]+=1
    eligible={split:raw_images_by_split.get(split,0)-held_by_split.get(split,0) for split in SPLITS}
    if any(count<0 for count in eligible.values()):raise ValueError('Held logical image count exceeds the verified source domain')
    return held_by_split,eligible

def build_index(manifest_path,output_dir,*,max_child_records=65536,max_child_metadata_mib=256,runtime_eligibility_registry=None):
    """Validate the entire frozen producer domain; stream parent, retain one bounded child.

    Child record/metadata limits are memory guards, not an assertion of global readiness.
    INDEX_COMPLETE is published only after all children, parent order/hash, identities,
    source components, subject splits and exact voxel cross-split checks pass.
    """
    from . import corpus,wan
    from .runtime_eligibility import RuntimeEligibility
    runtime=RuntimeEligibility(runtime_eligibility_registry)
    manifest=Path(manifest_path).resolve();output=Path(output_dir).absolute()
    if manifest.suffix!='.jsonl':raise ValueError('Explicit metadata JSONL required, never a tensor/MRI input')
    if output.exists():raise FileExistsError('Use a new immutable index output; no replay/replacement')
    if (type(max_child_records) is not int or max_child_records<1 or type(max_child_metadata_mib) not in (int,float) or
        not math.isfinite(max_child_metadata_mib) or max_child_metadata_mib<=0):
        raise ValueError('Positive explicit child metadata bounds required')
    tracked={}
    def track(path,digest):
        path=Path(path).absolute();value={'path':str(path),'sha256':digest,'fingerprint':_fingerprint(path)}
        old=tracked.get(str(path))
        if old is not None and old!=value:raise ValueError('Producer metadata changed or conflicting proof')
        tracked[str(path)]=value
    parent_contract_path=manifest.parent/'corpus_contract.json';parent_complete_path=manifest.parent/'CORPUS_COMPLETE'
    is_corpus=parent_contract_path.exists() or parent_complete_path.exists()
    parent=None;parent_marker=None
    if is_corpus:
        if not parent_contract_path.is_file() or not parent_complete_path.is_file():raise ValueError('Incomplete corpus producer')
        parent,parent_sha=_json_file(parent_contract_path);parent_marker,parent_marker_sha=_json_file(parent_complete_path)
        if (parent.get('schema_version')!=corpus.CORPUS_VERSION or parent.get('manifest_file')!=manifest.name or
            parent_marker.get('schema_version')!=corpus.CORPUS_COMPLETE_VERSION or parent_marker.get('contract_sha256')!=parent_sha or
            type(parent_marker.get('records')) is not int or parent_marker['records']<1):raise ValueError('Corpus contract/COMPLETE differs')
        entries=parent.get('children')
        if not isinstance(entries,list) or not entries or type(parent_marker.get('children')) is not int or parent_marker.get('children')!=len(entries):raise ValueError('Corpus child domain incomplete')
        track(parent_contract_path,parent_sha);track(parent_complete_path,parent_marker_sha)
        track(manifest,parent_marker['manifest_sha256'])
    else:
        entries=[None]
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent));os.chmod(stage,0o700)
    db=sqlite3.connect(stage/DB_NAME);_schema(db)
    counts=Counter();split_ordinals=Counter();image_counts=Counter();protocols={};common=None
    row_digest=hashlib.sha256();source_digest=hashlib.sha256();record_count=0;image_count=0;audited_images=0
    verified_entries=[];seen_children=set();published=False
    parent_stream=manifest.open('rb') if is_corpus else None
    def next_parent_row():
        for line in parent_stream:
            source_digest.update(line)
            if line.strip():return json.loads(line)
        return None
    try:
        for entry in entries:
            if is_corpus:
                if (not isinstance(entry,dict) or set(entry)!=corpus._CHILD_FIELDS or type(entry.get('records')) is not int or
                    not isinstance(entry.get('axes'),list) or any(type(a) is not int for a in entry['axes'])):raise ValueError('Corpus child proof schema differs')
                child_path=corpus._resolve_path(entry['manifest_path'],manifest.parent,'child manifest')
            else:child_path=manifest
            if child_path.suffix!='.jsonl' or child_path in seen_children:raise ValueError('Invalid or duplicate metadata child')
            seen_children.add(child_path)
            complete_path=child_path.parent/'COMPLETE';child_complete,complete_sha=_json_file(complete_path)
            child_records=child_complete.get('records')
            if type(child_records) is not int or not 0<child_records<=max_child_records:
                raise ValueError('Child exceeds --max-child-records; produce bounded complete encoding children')
            metadata_paths=[child_path,child_path.parent/'encoding_contract.json',complete_path]
            if 'used_images_sha256' in child_complete:metadata_paths.append(child_path.parent/'used_images.jsonl')
            if sum(p.stat().st_size for p in metadata_paths)>max_child_metadata_mib*1024**2:
                raise ValueError('Child exceeds --max-child-metadata-mib; no unbounded child parse')
            track(child_path,child_complete['manifest_sha256']);track(metadata_paths[1],child_complete['contract_sha256']);track(complete_path,complete_sha)
            if len(metadata_paths)>3:track(metadata_paths[3],child_complete['used_images_sha256'])
            child=corpus._verified_child(child_path,wan._load_single_manifest,audit_latents=False)
            if is_corpus:
                normalized=dict(entry)
                for name in ('manifest_path','encoding_contract_path','complete_path'):
                    normalized[name]=str(corpus._resolve_path(entry[name],manifest.parent,name))
                if normalized!=child['entry']:raise ValueError('Child differs from parent producer proof')
            verified_entries.append(child['entry'])
            rep=child['representation'];protocol=child['compatibility']
            if common is None:common={k:protocol.get(k) for k in ('vae_checkpoint_sha256','dtype')};common['latent_channels']=48
            if any(protocol.get(k)!=common[k] for k in ('vae_checkpoint_sha256','dtype')):raise ValueError('Incompatible child VAE/dtype')
            if rep in protocols and protocols[rep]!=protocol:raise ValueError('Representation geometry/raster/token protocol changed')
            protocols[rep]=protocol
            # SQL retains global evidence; only the current child's bounded groups stay in RAM.
            child_image_ids={}
            for identity,image in child['images'].items():
                metadata=image['metadata'];subject=(metadata['dataset'],metadata['subject_id']);split=metadata['split']
                runtime.reason(dict(metadata,dataset=identity[0],image_id=identity[1]))
                previous=db.execute('SELECT split FROM subjects WHERE dataset=? AND subject_id=?',subject).fetchone()
                if previous and previous[0]!=split:raise ValueError('Subject crosses train/val/test')
                db.execute('INSERT OR IGNORE INTO subjects VALUES(?,?,?)',(*subject,split))
                voxel=image.get('preprocessed_volume_sha256')
                if voxel is not None:
                    prior=db.execute('SELECT split FROM voxel_splits WHERE representation=? AND voxel_sha256=?',(rep,voxel)).fetchone()
                    if prior and prior[0]!=split:raise ValueError('Exact standardized voxel digest crosses splits')
                    db.execute('INSERT OR IGNORE INTO voxel_splits VALUES(?,?,?)',(rep,voxel,split));audited_images+=1
                image_idx=image_count;image_count+=1
                try:
                    db.execute('INSERT INTO images(image_index,dataset,image_id,subject_id,modality,split,representation,source_identity,voxel_sha256) VALUES(?,?,?,?,?,?,?,?,?)',
                        (image_idx,*identity,metadata['subject_id'],metadata['modality'],split,rep,encoded(image['source_identity']).decode(),voxel))
                except sqlite3.IntegrityError as error:raise ValueError('Duplicate canonical image or source container/component across children') from error
                child_image_ids[identity]=image_idx;image_counts[split]+=1
            for child_row in child['records']:
                row={**child_row,'source_encoding_manifest':child['entry']['manifest_path']} if is_corpus else child_row
                if is_corpus:
                    actual=next_parent_row()
                    if actual is None or encoded(actual)!=encoded(row):raise ValueError('Corpus row differs from complete child order/identity/split')
                payload=encoded(row);row_digest.update(payload)
                split=row['split'];local=split_ordinals[split];split_ordinals[split]+=1
                image_idx=child_image_ids[(row['dataset'],row['image_id'])];axis=row['axis'];shape=encoded(row['latent_shape']).decode()
                try:
                    db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)',(record_count,split,local,image_idx,axis,shape,row['latent_path'],row['latent_sha256'],payload.decode()))
                except sqlite3.IntegrityError as error:raise ValueError('Repeated latent path/hash/image axis in global producer') from error
                db.execute(f'UPDATE images SET first_record_index=COALESCE(first_record_index,?),axes_mask=axes_mask|?,record{axis}=?,splitrecord{axis}=?,shape{axis}=? WHERE image_index=?',
                           (record_count,1<<axis,record_count,local,shape,image_idx))
                record_count+=1;counts[row['dataset']+'/'+split]+=1
            db.commit();del child,child_image_ids
        if is_corpus:
            if next_parent_row() is not None:raise ValueError('Corpus contains rows outside completed child domain')
            if source_digest.hexdigest()!=parent_marker['manifest_sha256'] or record_count!=parent_marker['records']:
                raise ValueError('Parent corpus body hash/count changed')
            source_sha=source_digest.hexdigest()
        else:
            source_sha=verified_entries[0]['manifest_sha256'];track(manifest,source_sha)
        compatibility={'common':common,'representations':dict(sorted(protocols.items()))}
        subject_count=db.execute('SELECT count(*) FROM subjects').fetchone()[0]
        summary={'records':record_count,'source_images':image_count,'subjects':subject_count,
            'counts_by_dataset_split':dict(sorted(counts.items())),'compatibility':compatibility,
            'axis_coverage_policy':'each_image_matches_its_child_contract_axes','voxel_hash_audited_images':audited_images,
            'duplicate_screening':'source_container_component_and_cross_split_exact_preprocessed_voxel_sha256; no_fuzzy_matching'}
        if is_corpus:
            if any(type(parent.get(k)) is not int for k in ('records','source_images','subjects')):raise ValueError('Parent derived counts must be integers, not bools')
            comparison=dict(summary);stored=parent.get('compatibility')
            if isinstance(stored,dict) and 'common' not in stored:
                if any(p!=stored for p in protocols.values()):raise ValueError('Legacy corpus geometry differs')
                comparison['compatibility']=stored
            if any(parent.get(k)!=v for k,v in comparison.items()):raise ValueError('Parent derived counts/protocol do not match verified sources')
        for proof in tracked.values():
            if _fingerprint(proof['path'])!=proof['fingerprint']:raise ValueError('Producer metadata changed during index compilation')
        incomplete=dict(db.execute("SELECT split,count(*) FROM images WHERE representation='volume_xyz' AND axes_mask!=7 GROUP BY split"))
        runtime_summary={**summary,'records_by_split':dict(split_ordinals),'images_by_split':dict(image_counts),'incomplete_xyz_images_by_split':incomplete}
        held_images=runtime.excluded_image_indices(db)
        held_images_by_split,eligible_images_by_split=eligible_logical_image_counts(db,image_counts,held_images)
        held_records=Counter()
        for image_idx in held_images:
            for split,n in db.execute('SELECT split,count(*) FROM records WHERE image_index=? GROUP BY split',(image_idx,)):held_records[split]+=n
        runtime.assert_current()
        eligibility={'held_source_images':len(held_images),'held_records_by_split':dict(held_records),'eligible_records':record_count-sum(held_records.values()),'complete_source_domain_preserved':True,
            'held_source_images_by_split':dict(held_images_by_split),'eligible_source_images':sum(eligible_images_by_split.values()),
            'eligible_images_by_split':eligible_images_by_split,'logical_image_counting_unit':'one canonical MRI or native video, independent of axis records and bucket padding'}
        contract={'schema_version':VERSION,'source_manifest':str(manifest),'source_manifest_sha256':source_sha,
            'source_kind':'complete_corpus' if is_corpus else 'complete_encoding_child','source_metadata':list(tracked.values()),
            'children':verified_entries,'logical_rows_sha256':row_digest.hexdigest(),'summary':runtime_summary,
            'max_child_records':max_child_records,'max_child_metadata_mib':max_child_metadata_mib,
            'validation':'mature_verified_child_and_used_images;stream_parent_exact_order;SQL_global_integrity',
            'raw_MRI_or_latent_IO_or_stat_or_hash':False,'full_database_SHA_not_recomputed':True,
            'mandatory_runtime_eligibility':runtime.binding,'runtime_eligible_summary':eligibility}
        contract_sha=hashlib.sha256(encoded(contract)).hexdigest()
        for k,v in (('index_contract_sha256',contract_sha),('source_manifest_sha256',source_sha),('summary',runtime_summary),('logical_rows_sha256',row_digest.hexdigest()),('mandatory_runtime_eligibility',runtime.binding)):
            db.execute('INSERT INTO meta VALUES(?,?)',(k,encoded(v).decode()))
        db.commit();db.close()
        with (stage/DB_NAME).open('rb') as stream:os.fsync(stream.fileno())
        (stage/DB_NAME).chmod(0o444)
        _write(stage/'index_contract.json',contract);(stage/'index_contract.json').chmod(0o444)
        marker={'schema_version':VERSION,'index_contract_sha256':contract_sha,'source_manifest_sha256':source_sha,
            'logical_rows_sha256':row_digest.hexdigest(),'records':record_count,'images':image_count,'children':len(verified_entries),
            'database':{'name':DB_NAME,'fingerprint':_fingerprint(stage/DB_NAME)},'all_children_and_parent_complete':True,
            'mandatory_runtime_eligibility':runtime.binding,'runtime_eligible_summary':eligibility}
        _write(stage/MARKER,marker);(stage/MARKER).chmod(0o444)
        os.rename(stage,output);published=True
        return output
    finally:
        if parent_stream is not None:parent_stream.close()
        try:db.close()
        except sqlite3.Error:pass
        if not published:shutil.rmtree(stage,ignore_errors=True)

class _IndexHandle:
    def __init__(self,directory,runtime_eligibility_registry=None):
        from .runtime_eligibility import RuntimeEligibility
        self.runtime=RuntimeEligibility(runtime_eligibility_registry)
        self.directory=Path(directory).absolute();marker,_=_json_file(self.directory/MARKER);contract,contract_sha=_json_file(self.directory/'index_contract.json')
        if (marker.get('schema_version')!=VERSION or contract.get('schema_version')!=VERSION or marker.get('index_contract_sha256')!=contract_sha or
            marker.get('all_children_and_parent_complete') is not True or marker.get('logical_rows_sha256')!=contract.get('logical_rows_sha256') or
            marker.get('source_manifest_sha256')!=contract.get('source_manifest_sha256')):raise ValueError('Indexed input contract/COMPLETE mismatch')
        if marker.get('mandatory_runtime_eligibility')!=self.runtime.binding or contract.get('mandatory_runtime_eligibility')!=self.runtime.binding:
            raise ValueError('Index does not bind the mandatory current APPLIED runtime context; compile a new metadata index')
        self.database=self.directory/DB_NAME
        if marker.get('database',{}).get('name')!=DB_NAME or _fingerprint(self.database)!=marker['database']['fingerprint']:
            raise ValueError('Immutable index database fingerprint changed; no silent rebuild or DB rehash')
        for proof in contract['source_metadata']:
            if _fingerprint(proof['path'])!=proof['fingerprint']:raise ValueError('Frozen producer metadata changed since index compile')
        if (type(marker.get('records')) is not int or type(marker.get('images')) is not int or type(marker.get('children')) is not int or
            marker['records']!=contract['summary']['records'] or marker['images']!=contract['summary']['source_images'] or
            marker['children']!=len(contract['children'])):raise ValueError('Index COMPLETE derived counts differ from verified contract')
        self.summary=contract['summary'];self.expected_fp=marker['database']['fingerprint'];self._db=None
        self.binding={'index_directory':str(self.directory),'index_contract_sha256':contract_sha,
            'database_fingerprint':self.expected_fp,'source_manifest_sha256':contract['source_manifest_sha256'],
            'logical_rows_sha256':contract['logical_rows_sha256'],'schema_version':VERSION,
            'mandatory_runtime_eligibility':self.runtime.binding,'runtime_eligible_summary':contract['runtime_eligible_summary']}
        self.manifest_sha=contract['source_manifest_sha256']
        db=self.connection()
        for key,expected in (('index_contract_sha256',contract_sha),('source_manifest_sha256',self.manifest_sha),('summary',self.summary),('logical_rows_sha256',contract['logical_rows_sha256']),('mandatory_runtime_eligibility',self.runtime.binding)):
            value=db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
            if value is None or json.loads(value[0])!=expected:raise ValueError('Index SQL metadata does not bind completed producer')
        self.held_images=self.runtime.excluded_image_indices(db)
        self.held_images_by_split,self.eligible_images_by_split=eligible_logical_image_counts(db,self.summary['images_by_split'],self.held_images)
        derived={'held_source_images_by_split':dict(self.held_images_by_split),'eligible_source_images':sum(self.eligible_images_by_split.values()),'eligible_images_by_split':self.eligible_images_by_split}
        for key,value in derived.items():
            if key in contract['runtime_eligible_summary'] and contract['runtime_eligible_summary'][key]!=value:
                raise ValueError('Stored runtime logical image counts differ from actual held image grouping')
        self.held_records_by_split=Counter()
        for image_idx in self.held_images:
            for split,n in db.execute('SELECT split,count(*) FROM records WHERE image_index=? GROUP BY split',(image_idx,)):self.held_records_by_split[split]+=n
        self.eligible_incomplete=Counter()
        for image_idx,split in db.execute("SELECT image_index,split FROM images WHERE representation='volume_xyz' AND axes_mask!=7"):
            if image_idx not in self.held_images:self.eligible_incomplete[split]+=1
    def connection(self):
        if self._db is None:
            self._db=sqlite3.connect(self.database.as_uri()+'?mode=ro&immutable=1',uri=True)
            self._db.execute('PRAGMA query_only=ON');self._db.execute('PRAGMA cache_size=-8192')
        return self._db
    def close(self):
        if self._db is not None:self._db.close();self._db=None
        self.runtime.close()

class IndexedRecords(Sequence):
    """Parent-only lazy records. Worker queues still receive the mature small record dicts."""
    def __init__(self,handle,splits=None):
        self.handle=handle
        self.splits=None if splits is None or set(splits)==set(SPLITS) else tuple(s for s in SPLITS if s in splits)
        if self.splits is not None and not self.splits:raise ValueError('Nonempty train/val/test selection required')
    @property
    def binding(self):return self.handle.binding
    @property
    def manifest_sha256(self):return self.handle.manifest_sha
    @property
    def image_count(self):
        return self.handle.summary['source_images'] if self.splits is None else sum(self.handle.summary['images_by_split'].get(s,0) for s in self.splits)
    @property
    def eligible_image_count(self):
        relevant=SPLITS if self.splits is None else self.splits
        return sum(self.handle.eligible_images_by_split.get(split,0) for split in relevant)
    @property
    def eligible_count(self):
        relevant=SPLITS if self.splits is None else self.splits
        return len(self)-sum(self.handle.held_records_by_split.get(s,0) for s in relevant)
    def iter_eligible(self):
        self.handle.runtime.assert_current()
        for row in self:
            if self.handle.runtime.reason(row) is None:yield row
    def for_split(self,split):return self.for_splits((split,))
    def for_splits(self,splits):
        splits=tuple(splits)
        if any(s not in SPLITS for s in splits):raise ValueError('Invalid split')
        if self.splits is not None and not set(splits)<=set(self.splits):raise ValueError('Cannot expand a split-filtered view')
        return IndexedRecords(self.handle,splits)
    def __len__(self):
        return self.handle.summary['records'] if self.splits is None else sum(self.handle.summary['records_by_split'].get(s,0) for s in self.splits)
    def _where(self):
        return ('',()) if self.splits is None else (' WHERE split IN('+','.join('?' for _ in self.splits)+')',self.splits)
    def __iter__(self):
        where,args=self._where()
        for (raw,) in self.handle.connection().execute('SELECT record_json FROM records'+where+' ORDER BY record_index',args):yield json.loads(raw)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self))) ]
        if type(index) is not int:raise TypeError('Record index must be integer')
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if self.splits is None:query,args='SELECT record_json FROM records WHERE record_index=?',(index,)
        elif len(self.splits)==1:query,args='SELECT record_json FROM records WHERE split=? AND split_index=?',(self.splits[0],index)
        else:
            where,args=self._where();query='SELECT record_json FROM records'+where+' ORDER BY record_index LIMIT 1 OFFSET ?';args=(*args,index)
        row=self.handle.connection().execute(query,args).fetchone()
        if row is None:raise IndexError(index)
        return json.loads(row[0])
    def epoch_view_metadata(self,seed,epoch,axis_policy):
        """Yield only (view-local record index, shape), never materialize all record dicts."""
        from .data_loading import axis_for_image
        self.handle.runtime.assert_current()
        if type(epoch) is not int or epoch<0 or axis_policy not in ('one_per_image','all_axes'):raise ValueError('Invalid axis policy/epoch')
        if self.splits is not None and len(self.splits)>1:raise ValueError('Epoch sampler uses all records or one split; multi-split views are feature streams')
        where,args=self._where();db=self.handle.connection()
        if axis_policy=='all_axes':
            column='record_index' if self.splits is None else 'split_index'
            for index,shape,image_idx in db.execute('SELECT '+column+',shape,image_index FROM records'+where+' ORDER BY record_index',args):
                if image_idx not in self.handle.held_images:yield index,json.loads(shape)
            return
        relevant=SPLITS if self.splits is None else self.splits
        if any(self.handle.eligible_incomplete.get(s,0) for s in relevant):raise ValueError('Random-axis training requires all three cached axes per eligible MRI')
        columns='dataset,image_id,representation,record0,record1,record2,splitrecord0,splitrecord1,splitrecord2,shape0,shape1,shape2,image_index'
        for row in db.execute('SELECT '+columns+' FROM images'+where+' ORDER BY first_record_index',args):
            if row[12] in self.handle.held_images:continue
            axis=axis_for_image(row[0],row[1],seed,epoch) if row[2]=='volume_xyz' else 0
            offset=3 if self.splits is None else 6
            yield row[offset+axis],json.loads(row[9+axis])
    def close(self):self.handle.close()

def load_index(directory,*,runtime_eligibility_registry=None):
    handle=_IndexHandle(directory,runtime_eligibility_registry)
    return IndexedRecords(handle),handle.manifest_sha

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path,help='完整CORPUS JSONL或完整编码child JSONL；不是tensor路径')
    parser.add_argument('--output-dir',required=True,type=Path,help='新的不可变SQLite索引目录')
    parser.add_argument('--runtime-eligibility-registry',required=True,type=Path,help='Mandatory current APPLIED runtime eligibility registry')
    parser.add_argument('--max-child-records',type=int,default=65536,help='一次成熟child验证最多记录数；超过需生成完整小child，不代表全局已ready')
    parser.add_argument('--max-child-metadata-mib',type=float,default=256,help='每child manifest/contract/COMPLETE/used_images元数据总MiB上限；不读取latent测量')
    args=parser.parse_args(argv)
    path=build_index(args.manifest,args.output_dir,max_child_records=args.max_child_records,max_child_metadata_mib=args.max_child_metadata_mib,runtime_eligibility_registry=args.runtime_eligibility_registry)
    marker=json.loads((path/MARKER).read_text());print(json.dumps({'state':'INDEX_COMPLETE','directory':str(path),'records':marker['records'],'images':marker['images'],'children':marker['children'],'MRI_or_latent_IO':0}))

if __name__=='__main__':main()
