from __future__ import division
import os
import pandas as pd
import numpy as np
import openmatrix as omx
import yaml
import threading

class TableReader(threading.Thread):
    '''
    Multi-threaded table reader

    Parameters
    ----------
    tables (dict):
        Dictionary to put tables in
    names (list):
        List of table names
    fps (list):
        List of filepaths
    '''
    def __init__(self, tables, name, fp):
        threading.Thread.__init__(self)
        self.tables = tables
        self.name = name
        self.fp = fp

    def run(self):
        self.tables[self.name] = pd.read_csv(self.fp)

class crosstab:
    '''
    Crosstab summary
    '''
    def __init__(self, table, rows, columns, values, aggfunc, kwargs): #add aggfunc
        self.table = table
        self.rows = rows
        self.columns = columns
        self.values = values
        self.aggfunc = aggfunc
        self.kwargs = kwargs

    def run(self):
        return pd.crosstab(self.table[self.rows], self.table[self.columns], self.table[self.values], aggfunc = self.aggfunc, **self.kwargs).fillna(0)

class count:
    '''
    Counts summary
    '''
    def __init__(self, table, fields):
        self.table = table
        self.fields = fields

    def run(self):
        return self.table[self.fields].value_counts().sort_index().reset_index()

class aggregate:
    '''
    Aggregation summary
    '''
    def __init__(self, table, groupers, values):
        self.table = table
        if type(groupers) == list:
            self.groupers = groupers
        else:
            self.groupers = [groupers]
        self.values = values

    def run(self):
        return self.table[self.groupers + [self.values]].groupby(self.groupers).sum()

class Summary(threading.Thread):

    def __init__(self, filename, summary):
        threading.Thread.__init__(self)
        self.filename = filename
        self.summary = summary

    @classmethod
    def from_config(cls, location, info, tables):
        fp = location + '\\' + info['filepath']
        if info['type'] == 'crosstab':
            return cls(fp, crosstab(tables[info['table']], info['rows'], info['columns'], info['values'], info['aggfunc'], info['kwargs']))
        elif info['type'] == 'count':
            return cls(fp, count(tables[info['table']], info['fields']))
        elif info['type'] == 'aggregate':
            return cls(fp, aggregate(tables[info['table']], info['groupers'], info['values']))
        else:
            raise IOError('Summary type %s not supported'%s(info['type']))

    def run(self):
        self.summary.run().to_csv(self.filename)

if __name__ == '__main__':
    import time
    t0 = time.time()
    config_file = os.path.join(os.path.split(__file__)[0], 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    f.close()

    print('Reading ABM Outputs')
    abm_outputs = {}
    readers = []
    for name in config['abm_outputs']['tables']:
        readers.append(TableReader(abm_outputs, name, config['abm_outputs']['location'] + '\\' + config['abm_outputs']['tables'][name]))
        readers[-1].start()

    for i in range(len(readers)):
        readers[i].join()

    print('Creating Summaries')
    summaries = []
    for summary in config['summaries']:
        summaries.append(Summary.from_config(config['summary_location'], summary['summary'], abm_outputs))
        summaries[-1].start()

    for i in range(len(summaries)):
        summaries[i].join()

    t1 = time.time()
    print(t1 - t0)
    print('Go')

#if __name__ == '__main__':
#    df = pd.DataFrame({'a': ['a', 'b', 'c', 'a', 'b', 'c'], 'b': ['a', 'a', 'b', 'b', 'c', 'c'], 'c': range(6)})

#    summaries = []
#    summaries.append(Summary(r'C:\test\pipeline\crosstab.csv', crosstab(df, 'a', 'b', 'c', sum, {'margins': True, 'margins_name': 'Total'})))
#    summaries[-1].start()
#    summaries.append(Summary(r'C:\test\pipeline\share.csv', crosstab(df, 'a', 'b', 'c', sum, {'normalize': 'columns'})))
#    summaries[-1].start()
#    summaries.append(Summary(r'C:\test\pipeline\counts_by_a.csv', count({'table': df, 'fields': 'a'})))
#    summaries[-1].start()
#    summaries.append(Summary(r'C:\test\pipeline\counts_by_b.csv', count(df, 'b')))
#    summaries[-1].start()
#    summaries.append(Summary(r'C:\test\pipeline\counts_by_ab.csv', count(df, ['a', 'b'])))
#    summaries[-1].start()
#    summaries.append(Summary(r'C:\test\pipeline\agg_by_a.csv', aggregate(df, 'a', 'c')))
#    summaries[-1].start()
#    summaries.append(Summary(r'C:\test\pipeline\agg_by_b.csv', aggregate(df, 'b', 'c')))
#    summaries[-1].start()
#    summaries.append(Summary(r'C:\test\pipeline\agg_by_ab.csv', aggregate(df, ['a', 'b'], 'c')))
#    summaries[-1].start()

#    for i in range(len(summaries)):
#        summaries[i].join()

#    print('Done')