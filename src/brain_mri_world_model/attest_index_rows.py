"""Compare ordered immutable input-index metadata; never read latent/image bodies."""
import argparse,hashlib,json,pathlib,sqlite3
QUERY='''SELECT r.record_index,r.split,r.split_index,r.image_index,r.axis,r.shape,r.latent_sha256,
 i.dataset,i.image_id,i.subject_id,i.modality,i.representation,i.voxel_sha256
 FROM records r JOIN images i ON i.image_index=r.image_index ORDER BY r.record_index'''
def state(p):
 s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def compare(source,target):
 source=pathlib.Path(source);target=pathlib.Path(target);before=[state(source),state(target)];h=hashlib.sha256();count=0
 with sqlite3.connect(source.absolute().as_uri()+'?mode=ro&immutable=1',uri=True)as a,sqlite3.connect(target.absolute().as_uri()+'?mode=ro&immutable=1',uri=True)as b:
  ca=a.execute(QUERY);cb=b.execute(QUERY)
  while True:
   x=ca.fetchone();y=cb.fetchone()
   if x!=y:raise ValueError('SOURCE_TARGET_SEMANTIC_ORDER_OR_CONTENT_DIFFERENCE_AT_ROW:'+str(count))
   if x is None:break
   h.update((json.dumps(x,separators=(',',':'),ensure_ascii=False)+'\n').encode());count+=1
  def value(db,key):
   v=db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
   if not v:raise ValueError('REQUIRED_COMPLETE_INDEX_META_MISSING:'+key)
   try:return json.loads(v[0])
   except json.JSONDecodeError:return v[0]
  origin=value(a,'source_manifest_sha256');dest=value(b,'source_manifest_sha256')
 if before!=[state(source),state(target)]:raise ValueError('INPUT_INDEX_CHANGED_DURING_ATTESTATION')
 if not count:raise ValueError('EMPTY_INPUT_INDEX_NOT_ALLOWED')
 return {'state':'SOURCE_TARGET_SEMANTIC_INPUT_ORDER_IDENTICAL','records':count,'semantic_rows_sha256':h.hexdigest(),'source_manifest_sha256':origin,'target_manifest_sha256':dest,'source_index_file_state':before[0],'target_index_file_state':before[1],'latent_or_MRI_bodies_read':False}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-index',required=True);p.add_argument('--target-index',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=compare(a.source_index,a.target_index)
 with open(a.output,'x')as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
