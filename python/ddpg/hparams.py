import json
import re

# This class is partially compatible to tensorflows training.contrib.HParams class
class HParams:
  def __init__(self, **first_data):
    self.data = first_data

  def __str__(self):
    return str(self.data)

  def get_dict(self):
    return self.data

  def override_from_dict(self, new_data):
    for k,v in new_data:
      self.data[k] = v

  def items(self):
    return self.data.items()

  def values(self):
    data = {}

    def is_jsonable(x):
      try:
        json.dumps(x)
        return True
      except:
        return False

    for k,v in self.data.items():
      if(not is_jsonable):
        data[k] = str(v)
      else:
        data[k] = v

    return data

  def to_json(self):
    return json.dumps(self.values())

  def parse_json(self, args):
    if(not args):
      return

    data = json.loads(args)
    for key, value in data:
      self.data[key] = value

  # parses in the form of a=1,b=2,
  def parse(self, args):
    if(not args):
      return

    for line in re.compile("(\\w*?=(?:\\[.*?\\]|{.*?}|[^,])*)").split(args):
      if(not line or line == ','):
        continue
      [key, value] = line.split('=')
      try:
        value = eval(value) # unsafe as fuck
      except:
        pass
      self.data[key] = value

  def __getattr__(self, name):
    return self.data[name]