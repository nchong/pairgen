#!/usr/bin/env python

import getopt, sys
import yaml
from jinja2 import Environment, PackageLoader

def flatten(l):
  return [item for sublist in l for item in sublist]

def enum(*sequential, **named):
  enums = dict(zip(sequential, range(len(sequential))), **named)
  return type('Enum', (), enums)

class Constant:
  def __init__(self, name=None, description=None, type='double', arity=1):
    if not name:
      raise Exception, "New Constant requires name"
    self.name = name
    self.description = description
    self.type = type
    self.arity = arity

  def __repr__(self):
    return self.name

  def emit_name(self, name_prefix='', name_suffix=''):
    return name_prefix + self.name + name_suffix

  def device_name(self):
    return self.emit_name(name_prefix='d_')

  def sizeof(self):
    if self.arity == 1:
      return "sizeof(%s)" % self.type
    else:
      return "%d*sizeof(%s)" % (self.arity, self.type)

class Parameter:
  Set = enum('P', 'N')
  Access = enum('RO', 'RW', 'SUM')

  def __init__(self, name=None, type='double', arity=3,
               set=Set.P, access=Access.RO, reload=False, identity=None,
               description=None):
    if not name:
      raise Exception, "New Parameter requires name"
    if set == self.Set.N and access == self.Access.SUM:
      raise Exception, 'Per-neighbor sum parameter is not allowed'
    self.name = name
    self.type = type
    self.arity = arity
    self.set = set
    self.access = access
    self.reload = reload
    self.identity = identity
    self.description = description

  def __eq__(self, other):
    if self.name != other.name: return False
    if self.type != other.type: return False
    if self.arity != other.arity: return False
    if self.set != other.set: return False
    if self.access != other.access: return False
    if self.reload != other.reload: return False
    if self.identity != other.identity: return False
    if self.description != other.description: return False
    return True

  def __repr__(self):
    set_map = {self.Set.P: 'P',
               self.Set.N: 'N'}
    access_map = {self.Access.RO: 'RO',
                  self.Access.RW: 'RW',
                  self.Access.SUM: 'SUM'}
    try:
      return '%s(%s,%s)' % (self.name, set_map[self.set], access_map[self.access])
    except KeyError:
      return self.name

  def is_type(self, set, access):
    set_map = {'P': self.Set.P,
               'N': self.Set.N,
               '-': self.set}
    access_map = {'RO' : self.Access.RO,
                  'RW' : self.Access.RW,
                  'SUM': self.Access.SUM,
                  '-': self.access}
    try:
      return (self.set, self.access) == (set_map[set], access_map[access])
    except KeyError:
      raise Exception, 'Unknown set [%s] or access[%s]' % (set, access)

  def emit_name(self, name_prefix='', name_suffix=''):
    return name_prefix + self.name + name_suffix

  def emit_tagged_name(self):
    if self.is_type('P', 'RO'):
      return [self.emit_name(name_suffix='i'), 
              self.emit_name(name_suffix='j')]
    elif self.is_type('P', 'RW'):
      return [self.emit_name(name_suffix='i')]
    elif self.is_type('P', 'SUM'):
      return [self.emit_name(name_suffix='i_delta')]
    elif self.is_type('N', 'RO') or self.is_type('N', 'RW'):
      return [self.emit_name()]

  def emit_declaration(self, name_prefix='', name_suffix='', include_arity=True):
    name = self.emit_name(name_prefix, name_suffix)
    if self.arity > 1 and include_arity:
      return '%s %s[%d]' % (self.type, name, self.arity)
    else:
      return '%s %s' % (self.type, name)

  def emit_tagged_declaration(self):
    if self.is_type('P', 'RO'):
      return [self.emit_declaration(name_suffix='i'), 
              self.emit_declaration(name_suffix='j')]
    elif self.is_type('P', 'RW'):
      return [self.emit_declaration(name_suffix='i')]
    elif self.is_type('P', 'SUM'):
      return [self.emit_declaration(name_suffix='i_delta')]
    elif self.is_type('N', 'RO') or self.is_type('N', 'RW'):
      return [self.emit_declaration()]

  def emit_pointer_to_declaration(self, name_prefix='', name_suffix=''):
    return self.emit_declaration(name_prefix='*'+name_prefix, 
                                 name_suffix=name_suffix,
                                 include_arity=False)

  def emit_tagged_pointer_to_declaration(self):
    if self.is_type('P', 'RO'):
      return [self.emit_pointer_to_declaration(name_suffix='i'), 
              self.emit_pointer_to_declaration(name_suffix='j')]
    elif self.is_type('P', 'RW'):
      return [self.emit_pointer_to_declaration(name_suffix='i')]
    elif self.is_type('P', 'SUM'):
      return [self.emit_pointer_to_declaration(name_suffix='i_delta')]
    elif self.is_type('N', 'RO') or self.is_type('N', 'RW'):
      return [self.emit_pointer_to_declaration()]

  def emit_list_of_declaration(self):
    if self.is_type('P', 'RW') or self.is_type('P', 'SUM'):
      return self.emit_pointer_to_declaration(name_suffix='i_list')
    elif self.is_type('P', 'RO') or self.is_type('N', 'RO') or self.is_type('N', 'RW'):
      return self.emit_pointer_to_declaration(name_suffix='_list')
    else:
      raise Exception, 'Do not know how to emit for param type %s.' % self

  def emit_assignment_i(self, omit_index=False):
    result_string = ''
    if self.is_type('P', 'RO') or self.is_type('P', 'RW'):
      result_string = '%s' % self.emit_name(name_suffix='i')
    elif self.is_type('P', 'SUM'):
      result_string = '%s' % self.emit_name(name_suffix='i_delta')
    else:
      raise Exception, 'Do not know how to emit for param type %s.' % self
    if self.arity > 1 and not omit_index:
      return ['%s[%d]' % (result_string, k) for k in range(self.arity)]
    else:
      return [result_string]

  def emit_assignment_j(self):
    result_string = ''
    if self.is_type('P', 'RO'):
      result_string = '%s' % self.emit_name(name_suffix='j')
    elif self.is_type('N', 'RO') or self.is_type('N', 'RW'):
      result_string = '%s' % self.emit_name()
    else:
      raise Exception, 'Do not know how to emit for param type %s.' % self
    if self.arity > 1:
      return ['%s[%d]' % (result_string, k) for k in range(self.arity)]
    else:
      return [result_string]

  """
  Give the additive identity for the parameter such that x + id = id + x = x
  Can be overriden if identity is given in constructor.
  """
  def emit_identity(self):
    if self.identity:
      return self.identity 
    map = {'double' : '0.0f', 'int' : '0'}
    id = map[self.type]
    if self.arity > 1:
      return '{' + ', '.join([id]*self.arity) + '}'
    else:
      return id

  def as_kernel_declaration(self):
    if self.arity == 1 and \
       (self.access == self.Access.RW or self.access == self.Access.SUM):
      return self.emit_tagged_pointer_to_declaration()
    else:
      return self.emit_tagged_declaration()

  def as_kernel_call_parameter(self):
    if self.arity == 1 and \
       (self.access == self.Access.RW or self.access == self.Access.SUM):
      return [ "&"+x for x in self.emit_tagged_name() ]
    else:
      return self.emit_tagged_name()

  def device_name(self):
    return self.emit_name(name_prefix='d_')

  def sizeof(self):
    if self.set == self.Set.P:
      return "N*%d*sizeof(%s)" % (self.arity, self.type)
    elif self.set == self.Set.N:
      return "NSLOT*N*%d*sizeof(%s)" % (self.arity, self.type)
    else:
      raise Exception, "Unknown set"

  def sizeof_in_chars(self):
    map = { 'double':8, 'int':4, 'char':1 }
    try:
      return map[self.type]
    except KeyError:
      raise Exception, 'Unknown sizeof(%s) in chars' % (self.type)

def usage():
  print "%s <params.yml>" % sys.argv[0]

if __name__ == '__main__':
  if len(sys.argv) < 2:
    usage()
    exit(1)
  f = open(sys.argv[1], 'r')
  yaml_input = yaml.load(f)

  name = yaml_input['name']
  desc = yaml_input['description']
  params = []
  for p in yaml_input['parameters']:
    if p.has_key('set'):
      if p['set'] == 'P':
        p['set'] = Parameter.Set.P
      elif p['set'] == 'N':
        p['set'] = Parameter.Set.N
      else:
        print "unrecognized set [%s]" % (p['set'])
    if p.has_key('access'):
      if p['access'] == 'RO':
        p['access'] = Parameter.Access.RO
      elif p['access'] == 'RW':
        p['access'] = Parameter.Access.RW
      elif p['access'] == 'SUM':
        p['access'] = Parameter.Access.SUM
      else:
        print "unrecognized access [%s]" % (p['access'])
    params.append(Parameter(**p))
  kernel_params = flatten([p.as_kernel_declaration() for p in params])
  kernel_call_params = flatten([p.as_kernel_call_parameter() for p in params])
  consts = []
  for c in yaml_input['constants']:
    consts.append(Constant(**c))

  env = Environment(loader=PackageLoader('pairgen', 'templates'))

  template = env.get_template('constants.h')
  template.stream(name=name, consts=consts).dump(name+'_constants.h')

  template = env.get_template('particle.h')
  template.stream(name=name, params=params).dump(name+'_particle.h')

  template = env.get_template('pair_kernel.cu')
  template.stream(name=name,
    params=params,
    consts=consts, 
    kernel_params=kernel_params).dump(name+'_pair_kernel.cu')

  template = env.get_template('tpa_compute_kernel.cu')
  template.stream(name=name,
    params=params,
    consts=consts, 
    kernel_call_params=kernel_call_params,
    kernel_params=kernel_params).dump(name+'_tpa_compute_kernel.cu')

  template = env.get_template('bpa_compute_kernel.cu')
  template.stream(name=name,
    params=params,
    consts=consts, 
    kernel_call_params=kernel_call_params,
    kernel_params=kernel_params).dump(name+'_bpa_compute_kernel.cu')

  template = env.get_template('wrapper.h')
  template.stream(name=name,
    params=params).dump(name+'_wrapper.h')
