"""Whitelist top-level MRI technical JSON; unselected values are lexically skipped."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

TEXT_FIELDS=frozenset(('Modality','ImageType','SeriesDescription','ProtocolName','BodyPartExamined',
    'AnatomicRegion','AnatomicRegionSequence','ComplexImageComponent','MRAcquisitionType','MRAcquisition'))
GEOMETRY_FIELDS=frozenset(('PixelSpacing','SliceThickness','SpacingBetweenSlices',
    'ImageOrientationPatient','ImagePositionPatient','Rows','Columns','NumberOfFrames','AcquisitionMatrix'))
FIELDS=TEXT_FIELDS|GEOMETRY_FIELDS
REGION_KEYS=frozenset(('CodeMeaning','CodeValue','CodingSchemeDesignator','AnatomicRegion','AnatomicRegionSequence'))
NUMBER=re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')
DECODER=json.JSONDecoder()
BRAIN=re.compile(r'(?<![a-z0-9])(?:brain|head|cerebr\w*|encephal\w*|intracranial|cranium)(?![a-z0-9])|颅脑|头颅|全脑|脑部|头部',re.I)
NONBRAIN=re.compile(r'(?<![a-z0-9])(?:spine|spinal|lumbar|thoracic|cervical|abdomen|abdominal|pelvis|pelvic|knee|ankle|cardiac|chest|heart|breast)(?![a-z0-9])|脊柱|脊髓|腰椎|颈椎|胸椎|腹部|盆腔|膝关节',re.I)
SEG=re.compile(r'(?<![a-z0-9])(?:seg|segmentation|dseg|label|labels|mask|brainmask)(?![a-z0-9])|分割',re.I)
PROJECTION=re.compile(r'(?<![a-z0-9])(?:mip|minip|projection|proj)(?![a-z0-9])|maximum.?intensity|投影',re.I)


def _space(text,i):
    while i<len(text) and text[i].isspace():i+=1
    return i


def _string_end(text,i):
    if i>=len(text) or text[i]!='"':raise ValueError('Expected JSON string')
    i+=1
    while i<len(text):
        if text[i]=='"':return i+1
        if text[i]=='\\':
            i+=1
            if i>=len(text):raise ValueError('Truncated JSON escape')
            if text[i]=='u':
                if i+4>=len(text) or not all(c in '0123456789abcdefABCDEF' for c in text[i+1:i+5]):raise ValueError('Invalid Unicode escape')
                i+=4
            elif text[i] not in '"\\/bfnrt':raise ValueError('Invalid JSON escape')
        elif ord(text[i])<32:raise ValueError('Invalid JSON control character')
        i+=1
    raise ValueError('Truncated JSON string')


def _skip(text,i):
    i=_space(text,i)
    if i>=len(text):raise ValueError('Truncated JSON value')
    if text[i]=='"':return _string_end(text,i)
    if text[i] in '{[':
        opening=text[i];closing='}' if opening=='{' else ']';i=_space(text,i+1)
        if i<len(text) and text[i]==closing:return i+1
        while True:
            if opening=='{':
                i=_space(text,_string_end(text,i))
                if i>=len(text) or text[i]!=':':raise ValueError('Missing JSON colon')
                i+=1
            i=_space(text,_skip(text,i))
            if i<len(text) and text[i]==closing:return i+1
            if i>=len(text) or text[i]!=',':raise ValueError('Missing JSON delimiter')
            i=_space(text,i+1)
    for literal in ('true','false','null'):
        if text.startswith(literal,i):return i+len(literal)
    match=NUMBER.match(text,i)
    if match:return match.end()
    raise ValueError('Invalid JSON value')


def _region(value):
    if isinstance(value,str):return value
    if isinstance(value,list):return [_region(v) for v in value if isinstance(v,(str,dict,list))]
    if isinstance(value,dict):return {k:_region(v) for k,v in value.items() if k in REGION_KEYS}
    return None


def _numeric(value):
    if isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value):return value
    if isinstance(value,list):
        values=[_numeric(v) for v in value]
        return values if all(v is not None for v in values) else None
    return None


def extract(text):
    """Parse only selected values, never deserialize report/diagnosis/patient values."""
    i=_space(text,0);result={};seen=set()
    if i>=len(text) or text[i]!='{':raise ValueError('Technical sidecar must be a JSON object')
    i=_space(text,i+1)
    if i<len(text) and text[i]=='}':return result
    while True:
        end=_string_end(text,i);key=DECODER.raw_decode(text,i)[0];i=_space(text,end)
        if i>=len(text) or text[i]!=':':raise ValueError('Missing JSON key delimiter')
        i=_space(text,i+1);end=_skip(text,i)
        if key in FIELDS:
            if key in seen:raise ValueError('Duplicate selected technical key')
            value=json.loads(text[i:end]);seen.add(key)
            if key in GEOMETRY_FIELDS:value=_numeric(value)
            elif key in ('AnatomicRegion','AnatomicRegionSequence'):value=_region(value)
            elif isinstance(value,list):value=[v for v in value if isinstance(v,str)]
            elif not isinstance(value,str):value=None
            if value is not None:result[key]=value
        i=_space(text,end)
        if i<len(text) and text[i]=='}':
            if _space(text,i+1)!=len(text):raise ValueError('Trailing JSON data')
            return result
        if i>=len(text) or text[i]!=',':raise ValueError('Missing JSON member delimiter')
        i=_space(text,i+1)


def text_values(value):
    if isinstance(value,str):return [value]
    if isinstance(value,list):return [s for v in value for s in text_values(v)]
    if isinstance(value,dict):return [s for v in value.values() for s in text_values(v)]
    return []


def interpret(fields):
    anatomy=' '.join(s for key in ['BodyPartExamined','AnatomicRegion','AnatomicRegionSequence','SeriesDescription','ProtocolName'] for s in text_values(fields.get(key)))
    technical=' '.join(s for key in ['ImageType','SeriesDescription','ProtocolName'] for s in text_values(fields.get(key)))
    brain=bool(BRAIN.search(anatomy));nonbrain=bool(NONBRAIN.search(anatomy))
    return dict(anatomy_state='CONFLICT_BRAIN_NONBRAIN' if brain and nonbrain else 'EXPLICIT_NONBRAIN' if nonbrain else 'EXPLICIT_BRAIN_SOURCE_METADATA' if brain else 'NO_EXPLICIT_ANATOMY',
        MR_modality=fields.get('Modality','').upper()=='MR',modality_missing='Modality' not in fields,
        segmentation_or_mask=bool(SEG.search(technical)),projection=bool(PROJECTION.search(technical)),
        MRAcquisitionType=str(fields.get('MRAcquisitionType',fields.get('MRAcquisition',''))).upper(),
        pixel_confirmed=False,full_brain_coverage_confirmed=False)


def source_sidecar(source):
    source=Path(source);name=source.name
    if name.endswith('.nii.gz'):base=name[:-7]
    elif name.endswith('.nii'):base=name[:-4]
    else:raise ValueError('Only exact NIfTI-stem technical JSON is admitted')
    return source.with_name(base+'.json')


def read_once(source,max_bytes=16*1024**2):
    """One stem JSON attempt; oversized/unparseable never gains admission."""
    path=source_sidecar(source)
    try:
        stat=path.stat();before=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        with path.open('r',encoding='utf-8-sig') as stream:text=stream.read(max_bytes+1)
        if len(text)>max_bytes:return dict(status='PENDING_OVERSIZED_TECHNICAL_JSON',fields={},sidecar_path=str(path),body_reads=1,fingerprint=before)
        fields=extract(text);after=path.stat();signature=[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns]
        if before!=signature:return dict(status='PENDING_SIDECAR_CHANGED',fields={},sidecar_path=str(path),body_reads=1,fingerprint=before)
        return dict(status='TECHNICAL_WHITELIST_CACHED',fields=fields,interpretation=interpret(fields),sidecar_path=str(path),body_reads=1,fingerprint=before)
    except FileNotFoundError:
        return dict(status='PENDING_MISSING_TECHNICAL_JSON',fields={},sidecar_path=str(path),body_reads=0)
    except Exception as error:
        return dict(status='PENDING_TECHNICAL_JSON_ERROR',error_type=type(error).__name__,fields={},sidecar_path=str(path),body_reads=1)
