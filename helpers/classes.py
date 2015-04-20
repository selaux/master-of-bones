# coding=utf-8

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

def filter_by_classes(things, allowed_classes):
    new_things = filter(lambda x: x['class'] in allowed_classes, things)
    return list(new_things)
