# coding=utf-8

import vtk
from random import random

CLASSES = [
    {
        'name': 'Gazella',
        'short': 'GZ',
        'id': 1,
        'identifiers': [
            'Gaz'
        ]
    },
    {
        'name': 'Sheep - Wild',
        'short': 'SW',
        'id': 2,
        'identifiers': [
            'GÖ',
            'GSR'
        ]
    },
    {
        'name': 'Sheep - Domestic',
        'short': 'SD',
        'id': 3,
        'identifiers': [
            'Ma',
            'MiZ'
        ]
    }
]
UNKNOWN_CLASS = {
    'id': 0,
    'short': '?',
    'name': 'Unknown'
}

color_lookup = vtk.vtkLookupTable()
color_lookup.SetTableRange(0, 1)
color_lookup.SetHueRange(0, 1)
color_lookup.SetSaturationRange(1, 1)
color_lookup.SetValueRange(0.8, 0.8)
color_lookup.SetAlphaRange(1, 1)
color_lookup.Build()

def get_class(filename):
    for c in CLASSES:
        for ident in c['identifiers']:
            if filename.startswith(ident):
                return c['id'], c['name']
    return UNKNOWN_CLASS['id'], UNKNOWN_CLASS['name']

def get_class_short(id):
    for c in CLASSES:
        if c['id'] == id:
            return c['short']
    return UNKNOWN_CLASS['short']

def get_class_name(id):
    for c in CLASSES:
        if c['id'] == id:
            return c['name']
    return UNKNOWN_CLASS['name']

def simple_hash(string, modulo=64):
    hash_value = 0xcbf29ce484222325
    for b in string:
        hash_value = hash_value ^ ord(b)
        hash_value = int(hash_value) * 0x100000001b3
    return hash_value % modulo

def get_classed_color(id, label=None):
    class_map = {
        0: 0,
        1: 0.37,
        2: 0.675,
        3: 0.17
    }
    color = [0.0, 0.0, 0.0]
    if label is None:
        additional = random() * .1 - 0.05
    else:
        additional = (simple_hash(label, modulo=64) / 64.0) * .1 - 0.05

    seed = class_map[id] + additional
    color_lookup.GetColor(seed, color)
    return color

def filter_by_classes(things, allowed_classes):
    new_things = filter(lambda x: x['class'] in allowed_classes, things)
    return list(new_things)
