"""Playground: Config Schema Validator (v0.1)
Validates sample YAML configs against a draft schema defined inline.
"""
from __future__ import annotations
import argparse, json, os, sys, re

try:
    import yaml  # type: ignore
except ImportError:
    yaml=None

SCHEMA_VERSION=1

SCHEMA={
  'version': {'type':'int','required':True,'min':1},
  'case': {'type':'str','required':True,'enum':['cavity','channel','tgv','scalar']},
  'grid': {'type':'dict','required':True,'schema':{
      'nx': {'type':'int','min':4,'max':4096,'required':True},
      'ny': {'type':'int','min':4,'max':4096,'required':True}
  }},
  'time': {'type':'dict','required':True,'schema':{
      'dt': {'type':'float','min':1e-8,'max':1.0,'required':True},
      'steps': {'type':'int','min':1,'max':10_000_000,'required':True}
  }},
  'solver': {'type':'dict','required':True,'schema':{
      'coupling': {'type':'str','enum':['simple','piso'], 'required':True},
      'relax_u': {'type':'float','min':0.1,'max':1.0,'required':True},
      'relax_p': {'type':'float','min':0.1,'max':1.0,'required':True}
  }}
}

class ValidationError(Exception):
    pass

def validate(schema, data, path='root'):
    if not isinstance(schema, dict):
        raise ValidationError(f'Bad schema at {path}')
    for key, spec in schema.items():
        if spec.get('required') and key not in data:
            raise ValidationError(f'Missing required key {path}.{key}')
    for key, val in data.items():
        if key not in schema:
            raise ValidationError(f'Unknown key {path}.{key}')
        spec = schema[key]
        t = spec.get('type')
        if t=='int' and not isinstance(val,int):
            raise ValidationError(f'Type error {path}.{key} expected int got {type(val)}')
        if t=='float' and not isinstance(val,(int,float)):
            raise ValidationError(f'Type error {path}.{key} expected float got {type(val)}')
        if t=='str' and not isinstance(val,str):
            raise ValidationError(f'Type error {path}.{key} expected str')
        if t=='dict':
            if not isinstance(val,dict):
                raise ValidationError(f'Type error {path}.{key} expected dict')
            validate(spec.get('schema',{}), val, f'{path}.{key}')
        if 'enum' in spec and val not in spec['enum']:
            raise ValidationError(f'Enum error {path}.{key} {val} not in {spec["enum"]}')
        if 'min' in spec and isinstance(val,(int,float)) and val < spec['min']:
            raise ValidationError(f'Min violation {path}.{key} {val} < {spec["min"]}')
        if 'max' in spec and isinstance(val,(int,float)) and val > spec['max']:
            raise ValidationError(f'Max violation {path}.{key} {val} > {spec["max"]}')
    return True

def main():
    import argparse, textwrap
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', help='YAML config files to validate')
    args=ap.parse_args()
    if yaml is None:
        print(json.dumps({'schema_version':SCHEMA_VERSION,'error':'pyyaml not installed'}))
        sys.exit(1)
    if not args.config:
        print(json.dumps({'schema_version':SCHEMA_VERSION,'error':'No config files provided'}))
        sys.exit(1)
    report=[]
    for cfg_path in args.config:
        try:
            with open(cfg_path,'r') as f:
                data = yaml.safe_load(f)
            validate(SCHEMA,data)
            report.append({'file':cfg_path,'status':'ok'})
        except Exception as e:
            report.append({'file':cfg_path,'status':'fail','error':str(e)})
    print(json.dumps({'schema_version':SCHEMA_VERSION,'results':report}))

if __name__=='__main__':
    main()
