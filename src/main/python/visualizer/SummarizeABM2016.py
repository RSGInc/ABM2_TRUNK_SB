from __future__ import division
import os
import pandas as pd
import numpy as np
import openmatrix as omx
import threading
import time

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
    def __init__(self, table, name, fp):
        threading.Thread.__init__(self)
        self.tables = tables
        self.name = name
        self.fp = fp

    def run(self):
        self.tables[self.name] = pd.read_csv(self.fp)

def pivot(df, row, col, val = None, f = sum):
    if val is None:
        return df[[row, col]].reset_index().groupby([row, col]).count()['index'].reset_index().pivot(row, col, 'index').fillna(0)
    else:
        return df[[row, col, val]].groupby([row, col]).f()[val].reset_index().pivot(row, col, val).fillna(0)

def count(df, cols):
    return df[cols].value_counts().sort_index().reset_index().rename(columns = {0: 'freq'})

ABMOutputDir = r'C:\test\visualizer_conversion\output' #Copied from T:\ABM\ABM_FY19\model_runs\ABM2Plus\v1221\2016_1422new_sxu\output
geogXWalkDir = os.path.split(ABMOutputDir)[0]
SkimDir = r'T:\ABM\ABM_FY19\model_runs\ABM2Plus\v1221\2016_1422new_sxu\output'
vizOutputDir = r'C:\test\visualizer_conversion\visualizer\outputs\summaries\BUILD'#os.path.split(ABMOutputDir)[0] + r'\visualizer\outputs\summaries\BUILD'

input_files = {'hh':                    ABMOutputDir + r'\householdData_3.csv', #Replace 3 with MAX_ITER (once I find out how MAX_ITER is defined...)
               'per':                   ABMOutputDir + r'\personData_3.csv',
               'tours':                 ABMOutputDir + r'\indivTourData_3.csv',
               'trips':                 ABMOutputDir + r'\indivTripData_3.csv',
               'jtrips':                ABMOutputDir + r'\jointTripData_3.csv',
               'unique_joint_tours':    ABMOutputDir + r'\jointTourData_3.csv',
               'wsLoc':                 ABMOutputDir + r'\wsLocResults_3.csv',
               'aoResults':             ABMOutputDir + r'\aoResults.csv',
               'aoResults_Pre':         ABMOutputDir + r'\aoResults_Pre.csv',
               'visitor_trips':         ABMOutputDir + r'\visitorTrips.csv',
               'mazCorrespondence':     geogXWalkDir + r'\visualizer\data\geographicXwalk_PMSA.csv'
               }
skim_file = SkimDir + r'\traffic_skims_MD.omx'

purpose_map = {'Work': 1,
               'University': 2,
               'School': 3,
               'Escort': 4,
               'Shop': 5,
               'Maintenance': 6,
               'Eating Out': 7,
               'Visiting': 8,
               'Discretionary': 9,
               'Work-Based': 10,
               'work related': 10}

purpid2name = {}
for purpose in purpose_map:
    if purpose_map[purpose] not in purpid2name:
        purpid2name[purpose_map[purpose]] = purpose

ptype_map = {1: 'FT Worker',
             2: 'PT Worker',
             3: 'Univ Stud',
             4: 'Non Worker',
             5: 'Retiree',
             6: 'Driv Stud',
             7: 'NonDriv Stud',
             8: 'Pre-School'}

tour_comp_map = {1: 'All Adult', 2: 'All Children', 3: 'Mixed'}

t0 = time.time()
print('Reading ABM Outputs')
tables = {}
readers = []
for name in input_files:
    readers.append(TableReader(tables, name, input_files[name]))
    readers[-1].start()

for i in range(len(readers)):
    readers[i].join()

#Update individual tour and trip files so that tours have unique IDs
maxtours = tables['tours']['person_id'].value_counts().max() # Maximum number of tours a person makes in a day
tables['tours']['cummulative_tours'] = range(1, tables['tours'].shape[0] + 1)
tables['tours']['last_person'] = np.hstack(([0], tables['tours']['person_id'][:-1]))
tables['tours']['new_person'] = (tables['tours']['person_id']) != (tables['tours']['last_person'])
tables['tours']['to_subtract'] = tables['tours']['new_person'] * np.hstack(([0], tables['tours']['cummulative_tours'].iloc[:-1]))
for i in range(maxtours):
    tables['tours']['to_subtract'] = np.where(tables['tours']['to_subtract'] == 0,
                                              np.hstack(([0], tables['tours']['to_subtract'].iloc[:-1])),
                                              tables['tours']['to_subtract'])
tables['tours']['tour_id2'] = tables['tours']['cummulative_tours'] - tables['tours']['to_subtract']

maxtrips = tables['trips']['person_id'].value_counts().max() #Maximum number of trips a person makes in a day
tables['trips']['new_tour'] = (tables['trips']['orig_purpose'] == 'Home') | ((tables['trips']['tour_purpose'] == 'Work-Based') & (tables['trips']['orig_purpose'] == 'Work'))
tables['trips']['cummulative_tours'] = np.cumsum(tables['trips']['new_tour'])
tables['trips']['last_person'] = np.hstack(([0], tables['trips']['person_id'][:-1]))
tables['trips']['new_person'] = (tables['trips']['person_id']) != (tables['trips']['last_person'])
tables['trips']['to_subtract'] = tables['trips']['new_person'] * np.hstack(([0], tables['trips']['cummulative_tours'].iloc[:-1]))
for i in range(maxtrips):
    tables['trips']['to_subtract'] = np.where(tables['trips']['to_subtract'] == 0,
                                              np.hstack(([0], tables['trips']['to_subtract'].iloc[:-1])),
                                              tables['trips']['to_subtract'])
tables['trips']['tour_id2'] = tables['trips']['cummulative_tours'] - tables['trips']['to_subtract']
#tables['trips']['end_wb_tour'] = (tables['trips']['tour_purpose'] == 'Work-Based') | (tables['trips']['dest_purpose'] == 'Work')

##Update joint tour and trip files so that tours have unique IDs
#maxtours = tables['unique_joint_tours']['person_id'].value_counts().max() # Maximum number of tours a person makes in a day
#tables['unique_joint_tours']['cummulative_tours'] = range(tables['unique_joint_tours'].shape[0])
#tables['unique_joint_tours']['last_person'] = np.hstack(([0], tables['unique_joint_tours']['person_id'][:-1]))
#tables['unique_joint_tours']['new_person'] = (tables['unique_joint_tours']['person_id']) != (tables['unique_joint_tours']['last_person'])
#tables['unique_joint_tours']['to_subtract'] = tables['unique_joint_tours']['new_person'] * np.hstack(([0], tables['unique_joint_tours']['cummulative_tours'].iloc[:-1]))
#for i in range(maxtours):
#    tables['unique_joint_tours']['to_subtract'] = np.where(tables['unique_joint_tours']['to_subtract'] == 0,
#                                                           np.hstack(([0], tables['unique_joint_tours']['to_subtract'].iloc[:-1])),
#                                                           tables['unique_joint_tours']['to_subtract'])
#tables['unique_joint_tours']['tour_id2'] = tables['unique_joint_tours']['cummulative_tours'] - tables['unique_joint_tours']['to_subtract']

#maxtrips = tables['jtrips']['person_id'].value_counts().max() #Maximum number of trips a person makes in a day
#tables['jtrips']['new_tour'] = (tables['jtrips']['orig_purpose'] == 'Home') | ((tables['jtrips']['tour_purpose'] == 'Work-Based') & (tables['jtrips']['orig_purpose'] == 'Work'))
#tables['jtrips']['cummulative_tours'] = np.cumsum(tables['jtrips']['new_tour'])
#tables['jtrips']['last_person'] = np.hstack(([0], tables['jtrips']['person_id'][:-1]))
#tables['jtrips']['new_person'] = (tables['jtrips']['person_id']) != (tables['jtrips']['last_person'])
#tables['jtrips']['to_subtract'] = tables['jtrips']['new_person'] * np.hstack(([0], tables['jtrips']['cummulative_tours'].iloc[:-1]))
#for i in range(maxtrips):
#    tables['jtrips']['to_subtract'] = np.where(tables['jtrips']['to_subtract'] == 0,
#                                              np.hstack(([0], tables['jtrips']['to_subtract'].iloc[:-1])),
#                                              tables['jtrips']['to_subtract'])
#tables['jtrips']['tour_id2'] = tables['jtrips']['cummulative_tours'] - tables['jtrips']['to_subtract']

t1 = time.time()
print(t1 - t0)

print('Reading Skims')
skim = omx.open_file(skim_file)
skimLookup = pd.Series(skim.mapping('zone_number')).sort_values()
N = len(skimLookup)
#DST_SKM = np.concatenate((np.reshape(skim['MD_SOV_TR_H_DIST'], N**2), [-1]))
DST_SKM = np.reshape(skim['MD_SOV_TR_H_DIST'], N**2)

t2 = time.time()
print(t2 - t1)

# Prepare files for computing summary statistics
print('Preparing Data')
tables['aoResults']['HHVEH'] = np.minimum(tables['aoResults']['AO'], 4)
tables['aoResults_Pre']['HHVEH'] = np.minimum(tables['aoResults_Pre']['AO'], 4)
tables['hh']['HHVEH'] = np.minimum(tables['hh']['autos'], 4)

tables['hh']['VEH_NEWCAT'] = np.empty_like(tables['hh'].index)
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 0), 1,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 1) & (tables['hh']['AVs'] == 0), 2,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 1), 3,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 2) & (tables['hh']['AVs'] == 0), 4,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 2), 5,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 1) & (tables['hh']['AVs'] == 1), 6,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 3) & (tables['hh']['AVs'] == 0), 7,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 3), 8,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 2) & (tables['hh']['AVs'] == 1), 9,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 1) & (tables['hh']['AVs'] == 2), 10, tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 4) & (tables['hh']['AVs'] == 0), 11, tables['hh']['VEH_NEWCAT'])

tables['hh']['HHSIZ'] = tables['hh']['hh_id'].map(tables['per'][['hh_id', 'person_id']].groupby('hh_id').count()['person_id'])
tables['hh']['HHSIZE'] = np.minimum(tables['hh']['HHSIZ'], 5)
tables['hh']['ADULTS'] = tables['hh']['hh_id'].map(tables['per'][['hh_id', 'person_id', 'age']].query('age >= 18 and age < 99').groupby('hh_id').count()['person_id'])

tables['per']['PERTYPE'] = tables['per']['type'].map({'Full-time worker': 1,
                                                      'Part-time worker': 2,
                                                      'University student': 3,
                                                      'Non-worker': 4,
                                                      'Retired': 5,
                                                      'Student of driving age': 6,
                                                      'Student of non-driving age': 7,
                                                      'Child too young for school': 8})

tables['mazCorrespondence'] = tables['mazCorrespondence'].set_index('mgra')
tables['wsLoc']['HDISTRICT'] = tables['wsLoc']['HomeMGRA'].map(tables['mazCorrespondence']['pmsa'])
tables['wsLoc']['WDISTRICT'] = tables['wsLoc']['WorkLocation'].map(tables['mazCorrespondence']['pmsa'])
tables['wsLoc']['HHTAZ'] = tables['wsLoc']['HomeMGRA'].map(tables['mazCorrespondence']['taz'])
tables['wsLoc']['WTAZ'] = tables['wsLoc']['WorkLocation'].map(tables['mazCorrespondence']['taz']).fillna(0).astype(int)
tables['wsLoc']['STAZ'] = tables['wsLoc']['SchoolLocation'].map(tables['mazCorrespondence']['taz']).fillna(0).astype(int)

tables['wsLoc']['HHIndex'] = tables['wsLoc']['HHTAZ'].map(skimLookup).fillna(0).astype(int)
tables['wsLoc']['WIndex'] = tables['wsLoc']['WTAZ'].map(skimLookup).fillna(0).astype(int)
tables['wsLoc']['SIndex'] = tables['wsLoc']['STAZ'].map(skimLookup).fillna(0).astype(int)
#tables['wsLoc']['HHWIndex'] = np.where(tables['wsLoc']['WTAZ'] > 0, N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['WIndex'], -1)
#tables['wsLoc']['HHSIndex'] = np.where(tables['wsLoc']['STAZ'] > 0, N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['SIndex'], -1)

#tables['wsLoc']['WorkLocationDistance'] = DST_SKM[tables['wsLoc']['HHWIndex']]#.apply(lambda i: DST_SKM[i])
#tables['wsLoc']['SchoolLocationDistance'] = DST_SKM[tables['wsLoc']['HHSIndex']]#.apply(lambda i: DST_SKM[i])

tables['wsLoc']['WorkLocationDistance'] = DST_SKM[N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['WIndex']]
tables['wsLoc']['SchoolLocationDistance'] = DST_SKM[N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['SIndex']]

t3 = time.time()
print(t3 - t2)

print('Computing Summary Statistics')
tables['aoResults_Pre'][['HHVEH', 'HHID']].groupby('HHVEH').count()['HHID'].reset_index().to_csv(vizOutputDir + r'\autoOwnership_Pre.csv')
tables['aoResults'][['HHVEH', 'HHID']].groupby('HHVEH').count()['HHID'].reset_index().to_csv(vizOutputDir + r'\autoOwnership.csv')
tables['hh'][['AVs', 'hh_id']].groupby('AVs').count()['hh_id'].reset_index().to_csv(vizOutputDir + r'\autoOwnership_AV.csv')
tables['hh'][['VEH_NEWCAT', 'hh_id']].groupby('VEH_NEWCAT').count()['hh_id'].reset_index().to_csv(vizOutputDir + r'\autoOwnership_new.csv')

# Zero auto HHs by TAZ
tables['hh']['HHTAZ'] = tables['hh']['home_mgra'].map(tables['mazCorrespondence']['taz'])
tables['hh']['ZeroAutoWGT'] = np.where(tables['hh']['HHVEH'] == 0, 1, 0)
tables['hh']['ZeroAutoWGT'] = tables['hh']['ZeroAutoWGT'].fillna(0)
tables['hh'][['HHTAZ', 'ZeroAutoWGT']].groupby('HHTAZ').sum()['ZeroAutoWGT'].reset_index().to_csv(vizOutputDir + r'\zeroAutoByTaz.csv')

pertypeDistbn = pd.DataFrame({'freq': tables['per']['PERTYPE'].value_counts().sort_index()})#.reset_index()
pertypeDistbn.index.name = 'PERTYPE'
pertypeDistbn.reset_index().to_csv(vizOutputDir + r'\pertypeDistbn.csv')
tables['per'][['tele_choice', 'person_id']].groupby('tele_choice').count()['person_id'].reset_index().to_csv(vizOutputDir + r'\teleCommute_frequency.csv')
tables['hh'][['transponder', 'hh_id']].groupby('transponder').count()['hh_id'].reset_index().to_csv(vizOutputDir + r'\transponder_ownership.csv')

# Micro-mobility
micro_r1 = tables['trips']['micro_walkMode'].reset_index().groupby('micro_walkMode').count()['index']
micro_r2 = tables['trips']['micro_trnAcc'].reset_index().groupby('micro_trnAcc').count()['index']
micro_r3 = tables['trips']['micro_trnEgr'].reset_index().groupby('micro_trnEgr').count()['index']

micro_v1 = tables['visitor_trips']['micro_walkMode'].reset_index().groupby('micro_walkMode').count()['index']
micro_v2 = tables['visitor_trips']['micro_trnAcc'].reset_index().groupby('micro_trnAcc').count()['index']
micro_v3 = tables['visitor_trips']['micro_trnEgr'].reset_index().groupby('micro_trnEgr').count()['index']

micromobility_summary = micro_r1 + micro_r2 + micro_r3 + micro_v1 + micro_v2 + micro_v3
micromobility_summary.index.name = 'micro_mode'
pd.DataFrame({'trips': micromobility_summary}).reset_index().to_csv(vizOutputDir + r'\micormobility.csv')

# Mandatory DC
workers = tables['wsLoc'].query('WorkLocation > 0 and WorkLocation != 99999').dropna(subset = ['WorkLocationDistance'])
students = tables['wsLoc'].query('SchoolLocation > 0 and SchoolLocation != 88888').dropna(subset = ['SchoolLocationDistance'])

# Code distance bins
workers['distbin'] = np.where(workers['WorkLocationDistance'] % 1 == 0, workers['WorkLocationDistance'] + 1, np.minimum(np.ceil(workers['WorkLocationDistance']), 51)).astype(int)
students['distbin'] = np.where(students['SchoolLocationDistance'] % 1 == 0, students['SchoolLocationDistance'] + 1, np.minimum(np.ceil(students['SchoolLocationDistance']), 51)).astype(int)

# Create subsets for university and school students
univ = students.query('PersonType == 3')
schl = students.query('PersonType >= 6')

# Compute TLFDs by district and total
tlfd_work = workers.reset_index()[['distbin', 'HDISTRICT', 'index']].groupby(['distbin', 'HDISTRICT']).count()['index'].reset_index().pivot('distbin', 'HDISTRICT', 'index').fillna(0)
tlfd_univ = univ.reset_index()[['distbin', 'HDISTRICT', 'index']].groupby(['distbin', 'HDISTRICT']).count()['index'].reset_index().pivot('distbin', 'HDISTRICT', 'index').fillna(0)
tlfd_schl = schl.query('PersonType >= 6').reset_index()[['distbin', 'HDISTRICT', 'index']].groupby(['distbin', 'HDISTRICT']).count()['index'].reset_index().pivot('distbin', 'HDISTRICT', 'index').fillna(0)

for tlfd in [tlfd_work, tlfd_univ, tlfd_schl]:
    for i in tlfd.columns:
        tlfd['District_{}'.format(i)] = tlfd[i]
        del tlfd[i]
    tlfd['Total'] = tlfd.sum(1)

tlfd_work.to_csv(vizOutputDir + r'\workTLFD.csv')
tlfd_univ.to_csv(vizOutputDir + r'\univTLFD.csv')
tlfd_schl.to_csv(vizOutputDir + r'\schlTLFD.csv')

# Output avg trip lengths for visualizer
workTripLengths = workers[['HDISTRICT', 'WorkLocationDistance']].groupby('HDISTRICT').sum()['WorkLocationDistance'] / workers[['HDISTRICT', 'WorkLocationDistance']].groupby('HDISTRICT').count()['WorkLocationDistance']
workTripLengths.index.name = 'District'
workTripLengths.loc['Total'] = workers['WorkLocationDistance'].sum() / workers.shape[0]

univTripLengths = univ[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').sum()['SchoolLocationDistance'] / univ[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').count()['SchoolLocationDistance']
univTripLengths.loc['Total'] = univ['SchoolLocationDistance'].sum() / univ.shape[0]

schlTripLengths = schl[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').sum()['SchoolLocationDistance'] / schl[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').count()['SchoolLocationDistance']
schlTripLengths.loc['Total'] = schl['SchoolLocationDistance'].sum() / schl.shape[0]

pd.DataFrame({'Work': workTripLengths, 'Univ': univTripLengths, 'Schl': schlTripLengths}).to_csv(vizOutputDir + r'\mandTripLengths.csv')

# Work from home [for each district and total]
districtWorkers = tables['wsLoc'][['WorkLocation', 'HDISTRICT']].query('WorkLocation > 0').groupby('HDISTRICT').count()['WorkLocation']
districtWfh = tables['wsLoc'][['WorkLocation', 'HDISTRICT']].query('WorkLocation == 99999').groupby('HDISTRICT').count()['WorkLocation']
wfh_summary = pd.DataFrame({'Workers': districtWorkers, 'WFH': districtWfh})
wfh_summary.index.name = 'District'
wfh_summary.loc['Total'] = wfh_summary.sum(0)
wfh_summary.to_csv(vizOutputDir + r'\wfh_summary.csv')
totalwfh = pd.DataFrame(wfh_summary.loc['Total']).T
totalwfh.index.name = 'District'
totalwfh.to_csv(vizOutputDir + r'\wfh_summary_region.csv')

#County-County Flows
#JJF: Don't we only have one county? :)
countyFlows = workers[['HDISTRICT', 'WDISTRICT', 'PersonID']].groupby(['HDISTRICT', 'WDISTRICT']).count()['PersonID'].reset_index().pivot('HDISTRICT', 'WDISTRICT', 'PersonID').fillna(0)
countyFlows.loc['Total'] = countyFlows.sum(0)
countyFlows['Total'] = countyFlows.sum(1)
districts = countyFlows.columns
for i in range(2): #Rename the columns, then flip and rename the rows, then flip again
    for col in districts:
        try:
            countyFlows['District_{}'.format(int(col))] = countyFlows[col]            
        except ValueError:
            countyFlows['District_' + col] = countyFlows[col]
        del countyFlows[col]
    countyFlows = countyFlows.T
countyFlows.index.name = None
countyFlows.columns.name = None
countyFlows.to_csv(vizOutputDir + r'\countyFlows.csv')

t4 = time.time()
print(t4 - t3)

print('Processing Tour Files')
tables['tours']['PERTYPE'] = tables['tours']['person_type']
tables['tours']['DISTMILE'] = tables['tours']['tour_distance']
tables['tours']['HHVEH'] = tables['tours']['hh_id'].map(tables['hh'].set_index('hh_id')['HHVEH'])
tables['tours']['ADULTS'] = tables['tours']['hh_id'].map(tables['hh'].set_index('hh_id')['ADULTS'])
tables['tours']['AUTOSUFF'] = np.where(tables['tours']['HHVEH'] == 0, 0,
                                       np.where(tables['tours']['HHVEH'] < tables['tours']['ADULTS'], 1, 2))

tables['tours']['num_tot_stops'] = tables['tours']['num_ob_stops'] + tables['tours']['num_ib_stops']

tables['tours']['OTAZ'] = tables['tours']['orig_mgra'].map(tables['mazCorrespondence']['taz'])
tables['tours']['DTAZ'] = tables['tours']['dest_mgra'].map(tables['mazCorrespondence']['taz'])

tables['tours']['oindex'] = tables['tours']['OTAZ'].map(skimLookup)
tables['tours']['dindex'] = tables['tours']['DTAZ'].map(skimLookup)
tables['tours']['odindex'] = N*tables['tours']['oindex'] + tables['tours']['dindex']
tables['tours']['SKIMDIST'] = DST_SKM[N*tables['tours']['oindex'] + tables['tours']['dindex']]

tables['unique_joint_tours']['HHVEH'] = tables['unique_joint_tours']['hh_id'].map(tables['hh'].set_index('hh_id')['HHVEH'])
tables['unique_joint_tours']['ADULTS'] = tables['unique_joint_tours']['hh_id'].map(tables['hh'].set_index('hh_id')['ADULTS'])
tables['unique_joint_tours']['AUTOSUFF'] = np.where(tables['unique_joint_tours']['HHVEH'] == 0, 0,
                                                    np.where(tables['unique_joint_tours']['HHVEH'] < tables['unique_joint_tours']['ADULTS'], 1, 2))

#Code tour purposes
tables['tours']['TOURPURP'] = tables['tours']['tour_purpose'].map(purpose_map)
tables['tours']['TOURCAT'] = np.where(tables['tours']['TOURPURP'] <= 3, 0,
                                      np.where(tables['tours']['TOURPURP'] <= 9, 1, 2))

#Compute duration
tables['tours']['tourdur'] = tables['tours']['end_period'] - tables['tours']['start_period'] + 1 # to match survey
tables['tours']['TOURMODE'] = tables['tours']['tour_mode']

# exclude school escorting stop from ride sharing mandatory tours
tables['unique_joint_tours']['JOINT_PURP'] = tables['unique_joint_tours']['tour_purpose'].map(purpose_map)
tables['unique_joint_tours']['NUMBER_HH'] = (tables['unique_joint_tours']['tour_participants'].apply(len) + 1) // 2 # Number of people on tour

# get participant IDs and person types for each participant
#tables['per']['hhper_id'] = list(zip(tables['per']['hh_id'], tables['per']['person_num']))
tables['per']['hhper_id'] = tables['per']['hh_id'].astype(str) + '-' + tables['per']['person_num'].astype(str)
for i in range(1, 9):
    def get_perno(str):
        try:
            return str.replace(' ', '')[i-1]
        except IndexError:
            return 0
    tables['unique_joint_tours']['PER%d'%(i)] = tables['unique_joint_tours']['tour_participants'].apply(get_perno)
    tables['unique_joint_tours']['PTYPE%d'%(i)] = (tables['unique_joint_tours']['hh_id'].astype(str) + '-' + tables['unique_joint_tours']['PER%d'%(i)].astype(str)).map(tables['per'].set_index('hhper_id')['PERTYPE']).fillna(0)

tables['unique_joint_tours']['num_tot_stops'] = tables['unique_joint_tours']['num_ob_stops'] + tables['unique_joint_tours']['num_ib_stops']
tables['unique_joint_tours']['OTAZ'] = tables['unique_joint_tours']['orig_mgra'].map(tables['mazCorrespondence']['taz'])
tables['unique_joint_tours']['DTAZ'] = tables['unique_joint_tours']['dest_mgra'].map(tables['mazCorrespondence']['taz'])
tables['unique_joint_tours']['tourdur'] = tables['unique_joint_tours']['end_period'] - tables['unique_joint_tours']['start_period'] + 1
tables['unique_joint_tours']['TOURMODE'] = tables['unique_joint_tours']['tour_mode']

# ----
# this part is added by nagendra.dhakar@rsginc.com from binny.paul@rsginc.com soabm summaries
# translated from R to Python by joe.flood@sandag.org

# create a combined temp tour file for creating stop freq model summary
temp_tour1 = tables['tours'][['TOURPURP', 'num_ob_stops', 'num_ib_stops']]
temp_tour2 = tables['unique_joint_tours'][['JOINT_PURP', 'num_ob_stops', 'num_ib_stops']]
temp_tour2['TOURPURP'] = temp_tour2['JOINT_PURP']
del temp_tour2['JOINT_PURP']
temp_tour = pd.concat((temp_tour1, temp_tour2))

# code stop frequency model alternatives
temp_tour['STOP_FREQ_ALT'] = 4*np.minimum(temp_tour['num_ob_stops'], 3) + np.minimum(temp_tour['num_ib_stops'], 3) + 1
stopFreqModel_summary = pivot(temp_tour, 'STOP_FREQ_ALT', 'TOURPURP')
stopFreqModel_summary.index.name = None
stopFreqModel_summary.columns.name = None
stopFreqModel_summary.to_csv(vizOutputDir + r'\stopFreqModel_summary.csv')

# ------

t5 = time.time()
print(t5 - t4)

print('Processing trip files')
tables['trips']['TOURMODE'] = tables['trips']['tour_mode']
tables['trips']['TRIPMODE'] = tables['trips']['trip_mode']
tables['trips']['TOURPURP'] = tables['trips']['tour_purpose'].map(purpose_map)
tables['trips']['OPURP'] = tables['trips']['orig_purpose'].map(purpose_map)
tables['trips']['DPURP'] = tables['trips']['dest_purpose'].map(purpose_map)
tables['trips']['TOURCAT'] = np.where(tables['trips']['TOURPURP'] <= 3, 0,
                                      np.where(tables['trips']['TOURPURP'] <= 9, 1, 2))

#Mark stops and get other attributes
nr = tables['trips'].shape[0]
tables['trips']['inb_next'] = np.zeros_like(tables['trips'].index)
tables['trips']['inb_next'].iloc[:nr-1] = tables['trips']['inbound'][1:]
tables['trips']['stops'] = np.where((tables['trips']['DPURP'] > 0) & (((tables['trips']['inbound'] == 0) & (tables['trips']['inb_next'] == 0)) | ((tables['trips']['inbound'] == 1) & (tables['trips']['inb_next'] == 1))),
                                    1, 0)

tables['trips']['OTAZ'] = tables['trips']['orig_mgra'].map(tables['mazCorrespondence']['taz'])
tables['trips']['DTAZ'] = tables['trips']['dest_mgra'].map(tables['mazCorrespondence']['taz'])

cols = ['hh_id', 'person_num', 'TOURCAT', 'tour_id2']
coefs = [100000, 1000, 100, 1]
tables['tours']['lookup'] = tables['tours'][cols].dot(coefs)
tables['trips']['TOUROTAZ'] = tables['trips'][cols].dot(coefs).map(tables['tours'].set_index('lookup')['OTAZ'])
tables['trips']['TOURDTAZ'] = tables['trips'][cols].dot(coefs).map(tables['tours'].set_index('lookup')['DTAZ'])

tables['trips']['oindex'] = tables['trips']['OTAZ'].map(skimLookup)
tables['trips']['dindex'] = tables['trips']['DTAZ'].map(skimLookup)
tables['trips']['od_dist'] = DST_SKM[N*tables['trips']['oindex'] + tables['trips']['dindex']]

#create stops table
stops = tables['trips'].query('stops == 1')
stops['finaldestTAZ'] = np.where(stops['inbound'] == 0, stops['TOURDTAZ'], stops['TOUROTAZ'])

stops['oindex'] = stops['OTAZ'].map(skimLookup)
stops['dindex'] = stops['finaldestTAZ'].map(skimLookup)
stops['od_dist'] = DST_SKM[N*stops['oindex'] + stops['dindex']]

stops['oindex2'] = stops['OTAZ'].map(skimLookup)
stops['dindex2'] = stops['DTAZ'].map(skimLookup)
stops['os_dist'] = DST_SKM[N*stops['oindex2'] + stops['dindex2']]

stops['oindex3'] = stops['DTAZ'].map(skimLookup)
stops['dindex3'] = stops['finaldestTAZ'].map(skimLookup)
stops['sd_dist'] = DST_SKM[N*stops['oindex3'] + stops['dindex3']]

stops['out_dir_dist'] = stops['os_dist'] + stops['sd_dist'] - stops['od_dist']

tables['jtrips']['TOURMODE'] = tables['jtrips']['tour_mode']
tables['jtrips']['TRIPMODE'] = tables['jtrips']['trip_mode']
tables['jtrips']['TOURPURP'] = tables['jtrips']['tour_purpose'].map(purpose_map)
tables['jtrips']['OPURP'] = tables['jtrips']['orig_purpose'].map(purpose_map)
tables['jtrips']['DPURP'] = tables['jtrips']['dest_purpose'].map(purpose_map)
tables['jtrips']['TOURCAT'] = np.where(tables['jtrips']['TOURPURP'] <= 3, 0,
                                       np.where(tables['jtrips']['TOURPURP'] <= 9, 1, 2))

#Mark stops and get other attributes
nr = tables['jtrips'].shape[0]
tables['jtrips']['inb_next'] = np.zeros_like(tables['jtrips'].index)
tables['jtrips']['inb_next'].iloc[:nr-1] = tables['jtrips']['inbound'][1:]
tables['jtrips']['stops'] = np.where((tables['jtrips']['DPURP'] > 0) & (((tables['jtrips']['inbound'] == 0) & (tables['jtrips']['inb_next'] == 0)) | ((tables['jtrips']['inbound'] == 1) & (tables['jtrips']['inb_next'] == 1))),
                                     1, 0)

tables['jtrips']['OTAZ'] = tables['jtrips']['orig_mgra'].map(tables['mazCorrespondence']['taz'])
tables['jtrips']['DTAZ'] = tables['jtrips']['dest_mgra'].map(tables['mazCorrespondence']['taz'])

cols = ['hh_id', 'tour_id']
coefs = [10, 1]
tables['unique_joint_tours']['lookup'] = tables['unique_joint_tours'][cols].dot(coefs)
tables['jtrips']['TOUROTAZ'] = tables['jtrips'][cols].dot(coefs).map(tables['unique_joint_tours'].set_index('lookup')['OTAZ'])
tables['jtrips']['TOURDTAZ'] = tables['jtrips'][cols].dot(coefs).map(tables['unique_joint_tours'].set_index('lookup')['DTAZ'])

#create stops table
jstops = tables['jtrips'].query('stops == 1')
jstops['finaldestTAZ'] = np.where(jstops['inbound'] == 0, jstops['TOURDTAZ'], jstops['TOUROTAZ'])

jstops['oindex'] = jstops['OTAZ'].map(skimLookup)
jstops['dindex'] = jstops['finaldestTAZ'].map(skimLookup)
jstops['od_dist'] = DST_SKM[N*jstops['oindex'] + jstops['dindex']]

jstops['oindex2'] = jstops['OTAZ'].map(skimLookup)
jstops['dindex2'] = jstops['DTAZ'].map(skimLookup)
jstops['os_dist'] = DST_SKM[N*jstops['oindex2'] + jstops['dindex2']]

jstops['oindex3'] = jstops['DTAZ'].map(skimLookup)
jstops['dindex3'] = jstops['finaldestTAZ'].map(skimLookup)
jstops['sd_dist'] = DST_SKM[N*jstops['oindex3'] + jstops['dindex3']]

jstops['out_dir_dist'] = jstops['os_dist'] + jstops['sd_dist'] - jstops['od_dist']

t6 = time.time()
print(t6 - t5)

print('Calculating Tour Rates')
#workCounts = tables['tours'].reset_index()[['hh_id', 'person_num', 'TOURPURP', 'index']].query('TOURPURP == 1').groupby(['hh_id', 'person_num']).count()['index']
#schlCounts = tables['tours'].reset_index()[['hh_id', 'person_num', 'TOURPURP', 'index']].query('TOURPURP == 2 or TOURPURP == 3').groupby(['hh_id', 'person_num']).count()['index']
#inmCounts = tables['tours'].reset_index()[['hh_id', 'person_num', 'TOURPURP', 'index']].query('TOURPURP >= 4 and TOURPURP <= 9').groupby(['hh_id', 'person_num']).count()['index']
workCounts = tables['tours'].query('TOURPURP == 1')[['hh_id', 'person_num']].value_counts().sort_index()
schlCounts = tables['tours'].query('TOURPURP == 2 or TOURPURP == 3')[['hh_id', 'person_num']].value_counts().sort_index()
inmCounts = tables['tours'].query('TOURPURP >= 4 and TOURPURP <= 9')[['hh_id', 'person_num']].value_counts().sort_index()

#Individual NM tour generation
workCounts_temp = workCounts.copy()
schlCounts_temp = schlCounts.copy()
#inmCounts_temp = tables['tours'].reset_index()[['hh_id', 'person_num', 'TOURPURP', 'index']].query('TOURPURP > 4 and TOURPURP <= 9').groupby(['hh_id', 'person_num']).count()['index'].reset_index()
#atWorkCounts_temp = tables['tours'].reset_index()[['hh_id', 'person_num', 'TOURPURP', 'index']].query('TOURPURP == 10').groupby(['hh_id', 'person_num']).count()['index'].reset_index()
#escortCounts_temp = tables['tours'].reset_index()[['hh_id', 'person_num', 'TOURPURP', 'index']].query('TOURPURP == 4').groupby(['hh_id', 'person_num']).count()['index'].reset_index()
inmCounts_temp = tables['tours'].query('TOURPURP > 4 and TOURPURP <= 9')[['hh_id', 'person_num']].value_counts().sort_index()
atWorkCounts_temp = tables['tours'].query('TOURPURP == 10')[['hh_id', 'person_num']].value_counts().sort_index()
escortCounts_temp = tables['tours'].query('TOURPURP == 4')[['hh_id', 'person_num']].value_counts().sort_index()

temp = pd.DataFrame({'freq_work': workCounts_temp, 'freq_schl': schlCounts_temp}).fillna(0)
#temp1 = temp.copy()
#temp1['freq_inm'] = inmCounts_temp
temp1 = temp.merge(pd.DataFrame({'freq_inm': inmCounts_temp}).reset_index(), how = 'outer', on = ['hh_id', 'person_num']).fillna(0).sort_values(['hh_id', 'person_num'])
temp1['freq_m'] = temp1['freq_work'] + temp1['freq_schl']
temp1['freq_itours'] = temp1['freq_m'] + temp1['freq_inm']

#joint tours
#identify persons that made joint tour
temp_joint = pd.melt(tables['unique_joint_tours'], ['hh_id', 'tour_id'], ['PER1', 'PER2', 'PER3', 'PER4', 'PER5', 'PER6', 'PER7'], 'var', 'person_num')
temp_joint['person_num'] = temp_joint['person_num'].astype(int)
temp_joint['joint'] = np.where(temp_joint['person_num'] > 0, 1, 0)

temp_joint = temp_joint.query('joint == 1')
person_unique_joint = temp_joint.groupby(['hh_id', 'person_num']).sum()

temp2 = temp1.merge(person_unique_joint, how = 'outer', on = ['hh_id', 'person_num'])
temp2 = temp2.merge(pd.DataFrame({'freq_atwork': atWorkCounts_temp}).reset_index(), how = 'outer', on = ['hh_id', 'person_num'])
temp2 = temp2.merge(pd.DataFrame({'freq_escort': escortCounts_temp}).reset_index(), how = 'outer', on = ['hh_id', 'person_num'])
temp2 = temp2.fillna(0)

#add number of joint tours to non-mandatory
temp2['freq_nm'] = temp2['freq_inm'] + temp2['joint']

#get person type
tables['per']['lookup'] = (10*tables['per']['hh_id'] + tables['per']['person_num'])
temp2['PERTYPE'] = (10*temp2['hh_id'] + temp2['person_num']).map(tables['per'].set_index('lookup')['PERTYPE'])

#total tours
temp2['total_tours'] = temp2['freq_nm'] + temp2['freq_m'] + temp2['freq_atwork'] + temp2['freq_escort']

persons_mand = temp2.query('freq_m > 0') #persons with at least 1 mandatory tour
persons_nomand = temp2.query('freq_m == 0') #active persons with no mandatory tours

freq_nmtours_mand = pd.DataFrame({'freq': persons_mand[['PERTYPE', 'freq_nm']].value_counts()}).sort_index().reset_index()
freq_nmtours_nomand = pd.DataFrame({'freq': persons_nomand[['PERTYPE', 'freq_nm']].value_counts()}).sort_index().reset_index()
test = temp2[['PERTYPE', 'freq_inm', 'freq_m', 'freq_nm', 'freq_atwork', 'freq_escort']].value_counts().sort_index()

test.to_csv(vizOutputDir + r'\tour_rate_debug.csv')
del temp2['joint']
temp2.to_csv(vizOutputDir + r'\temp2.csv')

def write_table(df):
    text = str(df).replace(' ', ',')
    for i in range(df.shape[0]):
        text = text.replace('\n%d,'%(i), '\n')
    while ',,' in text:
        text = text.replace(',,', ',')
    return('\n'.join((text[1:].replace('\n,', '\n').split('\n'))[:-1]))

fp = vizOutputDir + r'\indivNMTourFreq.csv'
with open(fp, 'w') as f:
    f.write('x\nNon-Mandatory Tours for Persons with at-least 1 Mandatory Tour\n' + write_table(freq_nmtours_mand) + '\nx\nNon-Mandatory Tours for Active Persons with 0 Mandatory Tour\n' + write_table(freq_nmtours_nomand))
    f.close()

i4tourCounts = tables['tours'].query('TOURPURP == 4')[['hh_id', 'person_num']].value_counts()
i5tourCounts = tables['tours'].query('TOURPURP == 5')[['hh_id', 'person_num']].value_counts()
i6tourCounts = tables['tours'].query('TOURPURP == 6')[['hh_id', 'person_num']].value_counts()
i7tourCounts = tables['tours'].query('TOURPURP == 7')[['hh_id', 'person_num']].value_counts()
i8tourCounts = tables['tours'].query('TOURPURP == 8')[['hh_id', 'person_num']].value_counts()
i9tourCounts = tables['tours'].query('TOURPURP == 9')[['hh_id', 'person_num']].value_counts()
tourCounts = tables['tours'].query('TOURPURP <= 9')[['hh_id', 'person_num']].value_counts()
#dfs = [i4tourCounts, i5tourCounts, i6tourCounts, i7tourCounts, i8tourCounts, i9tourCounts, workCounts, schlCounts, inmCounts, tourCounts]

def counts2df(series):
    df = pd.DataFrame({'freq': series}).reset_index()
    df['lookup'] = 10*df['hh_id'] + df['person_num']
    df = df.set_index('lookup')
    return df

i4tourCounts = counts2df(i4tourCounts)
i5tourCounts = counts2df(i5tourCounts)
i6tourCounts = counts2df(i6tourCounts)
i7tourCounts = counts2df(i7tourCounts)
i8tourCounts = counts2df(i8tourCounts)
i9tourCounts = counts2df(i9tourCounts)
workCounts = counts2df(workCounts)
schlCounts = counts2df(schlCounts)
inmCounts = counts2df(inmCounts)
tourCounts = counts2df(tourCounts)

#for i in range(len(dfs)):
#    dfs[i] = pd.DataFrame({'freq': dfs[i]}).reset_index()
#    dfs[i]['lookup'] = 10*dfs[i]['hh_id'] + dfs[i]['person_num']
#    dfs[i] = dfs[i].set_index('lookup')

joint5 = tables['unique_joint_tours'].query('JOINT_PURP == 5')['hh_id'].value_counts()
joint6 = tables['unique_joint_tours'].query('JOINT_PURP == 6')['hh_id'].value_counts()
joint7 = tables['unique_joint_tours'].query('JOINT_PURP == 7')['hh_id'].value_counts()
joint8 = tables['unique_joint_tours'].query('JOINT_PURP == 8')['hh_id'].value_counts()
joint9 = tables['unique_joint_tours'].query('JOINT_PURP == 9')['hh_id'].value_counts()

tables['hh']['joint5'] = tables['hh']['hh_id'].map(joint5).fillna(0)
tables['hh']['joint6'] = tables['hh']['hh_id'].map(joint6).fillna(0)
tables['hh']['joint7'] = tables['hh']['hh_id'].map(joint7).fillna(0)
tables['hh']['joint8'] = tables['hh']['hh_id'].map(joint8).fillna(0)
tables['hh']['joint9'] = tables['hh']['hh_id'].map(joint9).fillna(0)
tables['hh']['jtours'] = tables['hh']['joint5'] + tables['hh']['joint6'] + tables['hh']['joint7'] + tables['hh']['joint8'] + tables['hh']['joint9']

#joint tour indicator
tables['hh']['JOINT'] = tables['hh']['cdap_pattern'].apply(lambda x: x[-1] == 'j').astype(int)

# code JTF category
tables['hh']['jtf'] = np.zeros_like(tables['hh'].index)
tables['hh']['jtf'] = np.where(tables['hh']['jtours'] == 0, 1, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint5'] == 1, 2, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint6'] == 1, 3, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint7'] == 1, 4, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint8'] == 1, 5, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint9'] == 1, 6, tables['hh']['jtf'])

tables['hh']['jtf'] = np.where(tables['hh']['joint5'] >= 2, 7, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint6'] >= 2, 8, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint7'] >= 2, 9, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint8'] >= 2, 10, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where(tables['hh']['joint9'] >= 2, 11, tables['hh']['jtf'])

tables['hh']['jtf'] = np.where((tables['hh']['joint5'] >= 1) & (tables['hh']['joint6'] >= 1), 12, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where((tables['hh']['joint5'] >= 1) & (tables['hh']['joint7'] >= 1), 13, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where((tables['hh']['joint5'] >= 1) & (tables['hh']['joint8'] >= 1), 14, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where((tables['hh']['joint5'] >= 1) & (tables['hh']['joint9'] >= 1), 15, tables['hh']['jtf'])

tables['hh']['jtf'] = np.where((tables['hh']['joint6'] >= 1) & (tables['hh']['joint7'] >= 1), 16, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where((tables['hh']['joint6'] >= 1) & (tables['hh']['joint8'] >= 1), 17, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where((tables['hh']['joint6'] >= 1) & (tables['hh']['joint9'] >= 1), 18, tables['hh']['jtf'])

tables['hh']['jtf'] = np.where((tables['hh']['joint7'] >= 1) & (tables['hh']['joint8'] >= 1), 19, tables['hh']['jtf'])
tables['hh']['jtf'] = np.where((tables['hh']['joint7'] >= 1) & (tables['hh']['joint9'] >= 1), 20, tables['hh']['jtf'])

tables['hh']['jtf'] = np.where((tables['hh']['joint8'] >= 1) & (tables['hh']['joint9'] >= 1), 21, tables['hh']['jtf'])

tables['per']['workTours'] = tables['per']['lookup'].map(workCounts['freq']).fillna(0)
tables['per']['schlTours'] = tables['per']['lookup'].map(schlCounts['freq']).fillna(0)
tables['per']['inmTours'] = tables['per']['lookup'].map(inmCounts['freq']).fillna(0)
tables['per']['numTours'] = tables['per']['lookup'].map(tourCounts['freq']).fillna(0)

tables['per']['i4numTours'] = tables['per']['lookup'].map(i4tourCounts['freq']).fillna(0)
tables['per']['i5numTours'] = tables['per']['lookup'].map(i5tourCounts['freq']).fillna(0)
tables['per']['i6numTours'] = tables['per']['lookup'].map(i6tourCounts['freq']).fillna(0)
tables['per']['i7numTours'] = tables['per']['lookup'].map(i7tourCounts['freq']).fillna(0)
tables['per']['i8numTours'] = tables['per']['lookup'].map(i8tourCounts['freq']).fillna(0)
tables['per']['i9numTours'] = tables['per']['lookup'].map(i9tourCounts['freq']).fillna(0)

# Total tours by person type
tables['per']['numTours'] = tables['per']['numTours'].fillna(0)
toursPertypeDistbn = pd.DataFrame({'freq': tables['tours'].query('PERTYPE > 0 and TOURPURP != 10')['PERTYPE'].value_counts().sort_index()})
toursPertypeDistbn.reset_index().to_csv(vizOutputDir + r'\toursPertypeDistbn.csv')

# count joint tour fr each person type
temp_joint = pd.melt(tables['unique_joint_tours'], ['hh_id', 'tour_id'], ['PTYPE1', 'PTYPE2', 'PTYPE3', 'PTYPE4', 'PTYPE5', 'PTYPE6', 'PTYPE7', 'PTYPE8'], value_name = 'PERTYPE')
#jtoursPertypeDistbn = temp_joint.groupby('PERTYPE').sum()['value']
jtoursPertypeDistbn = pd.DataFrame({'freq': temp_joint['PERTYPE'].astype(int).value_counts().sort_index()})
#jtoursPertypeDistbn2 = pd.DataFrame(np.empty((len(toursPertypeDistbn), 1)), toursPertypeDistbn.index, ['freq'])
#for i in jtoursPertypeDistbn2.index:
#    jtoursPertypeDistbn2.loc[i, 'freq'] = jtoursPertypeDistbn.loc['PTYPE%d'%(i)]

# Total tours by person type for visualizer
totaltoursPertypeDistbn = toursPertypeDistbn.copy()
totaltoursPertypeDistbn['freq'] += jtoursPertypeDistbn['freq']
totaltoursPertypeDistbn.to_csv(vizOutputDir + r'\total_tours_by_pertype_vis.csv', index = False)

# Total inidi NM tours by person type and purpose
tours_pertype_purpose = pd.DataFrame({'freq': tables['tours'].query('TOURPURP >= 4 and TOURPURP <= 9')[['PERTYPE', 'TOURPURP']].value_counts().sort_index()}).reset_index()
tours_pertype_purpose.to_csv(vizOutputDir + r'\tours_pertype_purpose.csv', index = False)

# code indi NM tour category
for i in range(4, 10):
    if i in [7, 8]:
        tables['per']['i%dnumTours'%(i)] = np.minimum(tables['per']['i%dnumTours'%(i)], 1)
    else:
        tables['per']['i%dnumTours'%(i)] = np.minimum(tables['per']['i%dnumTours'%(i)], 2)

tours_pertype_esco = tables['per'][['PERTYPE', 'i4numTours']].value_counts().sort_index().reset_index()
tours_pertype_shop = tables['per'][['PERTYPE', 'i5numTours']].value_counts().sort_index().reset_index()
tours_pertype_main = tables['per'][['PERTYPE', 'i6numTours']].value_counts().sort_index().reset_index()
tours_pertype_eati = tables['per'][['PERTYPE', 'i7numTours']].value_counts().sort_index().reset_index()
tours_pertype_visi = tables['per'][['PERTYPE', 'i8numTours']].value_counts().sort_index().reset_index()
tours_pertype_disc = tables['per'][['PERTYPE', 'i9numTours']].value_counts().sort_index().reset_index()

tours_pertype_esco.rename(columns = {'i4numTours': 'inumTours', 0: 'freq'}, inplace = True)
tours_pertype_shop.rename(columns = {'i5numTours': 'inumTours', 0: 'freq'}, inplace = True)
tours_pertype_main.rename(columns = {'i6numTours': 'inumTours', 0: 'freq'}, inplace = True)
tours_pertype_eati.rename(columns = {'i7numTours': 'inumTours', 0: 'freq'}, inplace = True)
tours_pertype_visi.rename(columns = {'i8numTours': 'inumTours', 0: 'freq'}, inplace = True)
tours_pertype_disc.rename(columns = {'i9numTours': 'inumTours', 0: 'freq'}, inplace = True)

tours_pertype_esco['purpose'] = 4*np.ones_like(tours_pertype_esco.index)
tours_pertype_shop['purpose'] = 5*np.ones_like(tours_pertype_shop.index)
tours_pertype_main['purpose'] = 6*np.ones_like(tours_pertype_main.index)
tours_pertype_eati['purpose'] = 7*np.ones_like(tours_pertype_eati.index)
tours_pertype_visi['purpose'] = 8*np.ones_like(tours_pertype_visi.index)
tours_pertype_disc['purpose'] = 9*np.ones_like(tours_pertype_disc.index)

indi_nm_tours_pertype = pd.concat((tours_pertype_esco, tours_pertype_shop, tours_pertype_main, tours_pertype_eati, tours_pertype_visi, tours_pertype_disc))
indi_nm_tours_pertype.to_csv(vizOutputDir + r'\inmtours_pertype_purpose.csv', index = False)

tours_pertype_purpose = tours_pertype_purpose.pivot('PERTYPE', 'TOURPURP', 'freq').fillna(0)

totalPersons = pertypeDistbn['freq'].sum()
pertypeDF = pertypeDistbn.copy()
pertypeDF.loc['All'] = totalPersons #Might need to make "Total" later
#nm_tour_rates = tours_pertype_purpose / pertypeDF['freq']
nm_tour_rates = tours_pertype_purpose.copy()
nm_tour_rates.loc['All'] = nm_tour_rates.sum(0)
for col in nm_tour_rates.columns:
    nm_tour_rates[col] /= pertypeDF['freq']
nm_tour_rates['All'] = nm_tour_rates.sum(1)
nm_tour_rates = nm_tour_rates.reset_index()
nm_tour_rates = pd.melt(nm_tour_rates, ['PERTYPE'], value_name =  'tour_rate')
nm_tour_rates = nm_tour_rates.rename(columns = {'TOURPURP': 'tour_purp'})
ptype_map['All'] = 'All'
purpid2name['All'] = 'All'
nm_tour_rates['PERTYPE'] = nm_tour_rates['PERTYPE'].map(ptype_map)
nm_tour_rates['tour_purp'] = nm_tour_rates['tour_purp'].map(purpid2name)
nm_tour_rates.to_csv(vizOutputDir + r'\nm_tours_rates.csv', index = False)

t1 = tables['tours'].query('TOURPURP < 10')['TOURPURP'].value_counts().sort_index()
t3 = tables['unique_joint_tours']['JOINT_PURP'].value_counts().sort_index()
tours_purpose_type = pd.DataFrame({'indi': t1, 'joint': t3}).fillna(0)
tours_purpose_type.to_csv(vizOutputDir + r'\tours_purpose_type.csv')

t7 = time.time()
print(t7 - t6)

print('Summarizing Daily Activity Patterns')
#per['activity_pattern'] = np.where(per['activity_pattern'] == 'M' & per['imf_choice'] == 0 & per['inmf_choice'] > 0, 'N', per['activity_pattern'])
tables['per']['activity_pattern'] = np.where((tables['per']['activity_pattern'] == 'M') & (tables['per']['imf_choice'] == 0),
                                             np.where(tables['per']['inmf_choice'] > 0, 'N', 'H'),
                                             tables['per']['activity_pattern'])
#dapSummary = tables['per'][['PERTYPE', 'activity_pattern']].value_counts().sort_index().reset_index().rename(columns = {0: 'freq'})
dapSummary = count(tables['per'], ['PERTYPE', 'activity_pattern'])
dapSummary.to_csv(vizOutputDir + r'\dapSummary.csv')

dapSummary_vis = dapSummary.pivot('PERTYPE', 'activity_pattern', 'freq').fillna(0)
dapSummary_vis.loc['Total'] = dapSummary_vis.sum(0)
dapSummary_vis = pd.melt(dapSummary_vis.reset_index(), id_vars = ['PERTYPE'], value_name = 'freq').rename(columns = {'activity_pattern': 'DAP'})
dapSummary_vis.to_csv(vizOutputDir + r'\dapSummary_vis.csv')

#hhsizeJoint = tables['hh'].query('HHSIZE >= 2')[['HHSIZE', 'JOINT']].value_counts().sort_index().reset_index().rename(columns = {0: 'freq'})
hhsizeJoint = count(tables['hh'].query('HHSIZE >= 2'), ['HHSIZE', 'JOINT'])
hhsizeJoint.to_csv(vizOutputDir + r'\hhsizeJoint.csv')

#mtfSummary = tables['per'].query('imf_choice > 0')[['PERTYPE', 'imf_choice']].value_counts().sort_index().reset_index().rename(columns = {0: 'freq'})
mtfSummary = count(tables['per'].query('imf_choice > 0'), ['PERTYPE', 'imf_choice'])
mtfSummary.to_csv(vizOutputDir + r'\mtfSummary.csv')

mtfSummary_vis = mtfSummary.pivot('PERTYPE', 'imf_choice', 'freq').fillna(0)
mtfSummary_vis.loc['Total'] = mtfSummary_vis.sum(0)
mtfSummary_vis = pd.melt(mtfSummary_vis.reset_index(), id_vars = ['PERTYPE'], value_name = 'freq').rename(columns = {'imf_choice': 'MTF'})
mtfSummary_vis.to_csv(vizOutputDir + r'\mtfSummary_vis.csv')

inmSummary = pd.DataFrame({'PERTYPE': range(1, 9)})
inmSummary['tour0'] = inmSummary['PERTYPE'].map(tables['per'].query('inmTours == 0')['PERTYPE'].value_counts())
inmSummary['tour1'] = inmSummary['PERTYPE'].map(tables['per'].query('inmTours == 1')['PERTYPE'].value_counts())
inmSummary['tour2'] = inmSummary['PERTYPE'].map(tables['per'].query('inmTours == 2')['PERTYPE'].value_counts())
inmSummary['tour3pl'] = inmSummary['PERTYPE'].map(tables['per'].query('inmTours >= 3')['PERTYPE'].value_counts())
inmSummary.to_csv(vizOutputDir + r'\innmSummary.csv')

inmSummary_viz = inmSummary.set_index('PERTYPE').rename(columns = {'tour0': '0', 'tour1': '1', 'tour2': '2', 'tour3pl': '3pl'})
inmSummary_viz.loc['Total'] = inmSummary_viz.sum(0)
inmSummary_viz = pd.melt(inmSummary_viz.reset_index(), id_vars = ['PERTYPE'], var_name = 'nmtours', value_name = 'freq')
inmSummary_viz.to_csv(vizOutputDir + r'\inmSummary_vis.csv')

#Joint Tour Frequency and composition
jtfSummary = tables['hh'].dropna(subset = ['jtf'])['jtf'].value_counts().sort_index().reset_index().rename(columns = {'index': 'jtf', 'jtf': 'freq'})
jointComp = tables['unique_joint_tours']['tour_composition'].value_counts().sort_index().reset_index().rename(columns = {'index': 'tour_composition', 'tour_composition': 'freq'})
jointPartySize = tables['unique_joint_tours']['NUMBER_HH'].value_counts().sort_index().reset_index().rename(columns = {'index': 'NUMBER_HH', 'NUMBER_HH': 'freq'})
jointCompPartySize = tables['unique_joint_tours'][['tour_composition', 'NUMBER_HH']].value_counts().sort_index().reset_index().rename(columns = {0: 'freq'})
tables['hh']['jointCat'] = np.minimum(tables['hh']['jtours'], 2)
jointToursHHSize = tables['hh'].dropna(subset = ['HHSIZE', 'jointCat'])[['HHSIZE', 'jointCat']].value_counts().sort_index().reset_index().rename(columns = {0: 'freq'})

jftSummary_file = vizOutputDir + r'\jtfSummary.csv'
jtfSummary.to_csv(jftSummary_file)
jointComp.to_csv(jftSummary_file, mode = 'a')
jointPartySize.to_csv(jftSummary_file, mode = 'a')
jointCompPartySize.to_csv(jftSummary_file, mode = 'a')
jointToursHHSize.to_csv(jftSummary_file, mode = 'a')

#cap joint party size to 5+
jointPartySize = np.minimum(tables['unique_joint_tours']['NUMBER_HH'], 5).value_counts().sort_index().reset_index().rename(columns = {'index': 'NUMBER_HH', 'NUMBER_HH': 'freq'}).query('NUMBER_HH <= 5')

jtf = pd.DataFrame({'jtf_code': range(1, 22),
                    'alt_name': ["No Joint Tours", "1 Shopping", "1 Maintenance", "1 Eating Out", "1 Visiting", "1 Other Discretionary",
                                 "2 Shopping", "1 Shopping / 1 Maintenance", "1 Shopping / 1 Eating Out", "1 Shopping / 1 Visiting",
                                 "1 Shopping / 1 Other Discretionary", "2 Maintenance", "1 Maintenance / 1 Eating Out",
                                 "1 Maintenance / 1 Visiting", "1 Maintenance / 1 Other Discretionary", "2 Eating Out", "1 Eating Out / 1 Visiting",
                                 "1 Eating Out / 1 Other Discretionary", "2 Visiting", "1 Visiting / 1 Other Discretionary", "2 Other Discretionary"]})
jtf['freq'] = jtf['jtf_code'].map(jtfSummary.set_index('jtf')['freq']).fillna(0)

jointComp['tour_composition'] = jointComp['tour_composition'].map(tour_comp_map)

#jointToursHHSizeProp = jointToursHHSize.query('HHSIZE > 1').pivot('jointCat', 'HHSIZE', 'freq')
jointToursHHSizeProp = pd.crosstab(jointToursHHSize.query('HHSIZE > 1')['jointCat'],
                                   jointToursHHSize.query('HHSIZE > 1')['HHSIZE'],
                                   jointToursHHSize.query('HHSIZE > 1')['freq'], aggfunc = sum,
                                   margins = True, margins_name = 'Total', normalize = 'columns').fillna(0)
jointToursHHSizeProp = pd.melt((100*jointToursHHSizeProp).reset_index(), id_vars = ['jointCat'], value_name = 'freq').rename(columns = {'jointCat': 'jointTours', 'HHSIZE': 'hhsize'})

jointCompPartySize['tour_composition'] = jointCompPartySize['tour_composition'].map(tour_comp_map)

#This calculation is not matching RSG's outputs--need to investigate why (Totals in RSG's outputs don't add up to 100%)
jointCompPartySizeProp = pd.crosstab(jointCompPartySize['tour_composition'],
                                     np.minimum(jointCompPartySize['NUMBER_HH'], 5),
                                     jointCompPartySize['freq'],aggfunc = sum,
                                     margins = True, margins_name = 'Total', normalize = 'index').fillna(0)
jointCompPartySizeProp = pd.melt((100*jointCompPartySizeProp).reset_index(), id_vars = ['tour_composition'], value_name = 'freq').rename(columns = {'tour_composition': 'comp', 'NUMBER_HH': 'partysize'})

#jointCompPartySizeProp
jtf.to_csv(vizOutputDir + r'\jtf.csv', index = False)
jointComp.to_csv(vizOutputDir + r'\jointComp.csv', index = False)
jointPartySize.to_csv(vizOutputDir + r'\JointPartySize.csv', index = False)
jointCompPartySizeProp.to_csv(vizOutputDir + r'\JointCompPartySize.csv', index = False)
jointToursHHSizeProp.to_csv(vizOutputDir + r'\JointToursHHSize.csv', index = False)

t8 = time.time()
print(t8 - t7)

print('Summarizing TOD Profiles')
tod = pd.crosstab(tables['tours']['start_period'], tables['tours']['TOURPURP'])
jtod = pd.crosstab(tables['unique_joint_tours']['start_period'], tables['unique_joint_tours']['JOINT_PURP'])
todDepProfile = pd.DataFrame(index = tod.index)
todDepProfile['work'] = tod[1]
todDepProfile['univ'] = tod[2]
todDepProfile['sch'] = tod[3]
todDepProfile['esc'] = tod[4]
todDepProfile['imain'] = tod[5] + tod[6]
todDepProfile['idisc'] = tod[7] + tod[8] + tod[9]
todDepProfile['jmain'] = jtod[5] + jtod[6]
todDepProfile['jdisc'] = jtod[7] + jtod[8] + jtod[9]
todDepProfile['atwork'] = tod[10]
todDepProfile['Total'] = todDepProfile.sum(1)
todDepProfile.fillna(0).to_csv(vizOutputDir + r'\todDepProfile.csv')

tod = pd.crosstab(tables['tours']['end_period'], tables['tours']['TOURPURP'])
jtod = pd.crosstab(tables['unique_joint_tours']['end_period'], tables['unique_joint_tours']['JOINT_PURP'])
todArrProfile = pd.DataFrame(index = tod.index)
todArrProfile['work'] = tod[1]
todArrProfile['univ'] = tod[2]
todArrProfile['sch'] = tod[3]
todArrProfile['esc'] = tod[4]
todArrProfile['imain'] = tod[5] + tod[6]
todArrProfile['idisc'] = tod[7] + tod[8] + tod[9]
todArrProfile['jmain'] = jtod[5] + jtod[6]
todArrProfile['jdisc'] = jtod[7] + jtod[8] + jtod[9]
todArrProfile['atwork'] = tod[10]
todArrProfile['Total'] = todArrProfile.sum(1)
todArrProfile.fillna(0).to_csv(vizOutputDir + r'\todArrProfile.csv')

tod = pd.crosstab(tables['tours']['tourdur'], tables['tours']['TOURPURP'])
jtod = pd.crosstab(tables['unique_joint_tours']['tourdur'], tables['unique_joint_tours']['JOINT_PURP'])
todDurProfile = pd.DataFrame(index = tod.index)
todDurProfile['work'] = tod[1]
todDurProfile['univ'] = tod[2]
todDurProfile['sch'] = tod[3]
todDurProfile['esc'] = tod[4]
todDurProfile['imain'] = tod[5] + tod[6]
todDurProfile['idisc'] = tod[7] + tod[8] + tod[9]
todDurProfile['jmain'] = jtod[5] + jtod[6]
todDurProfile['jdisc'] = jtod[7] + jtod[8] + jtod[9]
todDurProfile['atwork'] = tod[10]
todDurProfile['Total'] = todDurProfile.sum(1)
todDurProfile.fillna(0).to_csv(vizOutputDir + r'\todDurProfile.csv')

##stops by direction, purpose and model tod
def classify_tod(period):
    if period >= 4 and period <= 9:
        return 1
    if period >= 10 and period <= 22:
        return 2
    if period >= 23 and period <= 29:
        return 3
    if period >= 30 and period <= 40:
        return 4
    else:
        return 5

tables['tours']['start_tod'] = tables['tours']['start_period'].apply(classify_tod)
tables['tours']['end_tod'] = tables['tours']['end_period'].apply(classify_tod)
tables['unique_joint_tours']['start_tod'] = tables['unique_joint_tours']['start_period'].apply(classify_tod)
tables['unique_joint_tours']['end_tod'] = tables['unique_joint_tours']['end_period'].apply(classify_tod)

stops_ib_tod = tables['tours'][['num_ib_stops', 'tour_purpose', 'start_tod', 'end_tod']].groupby(['tour_purpose', 'start_tod', 'end_tod']).sum().reset_index().sort_values(['end_tod', 'start_tod', 'tour_purpose'])
stops_ob_tod = tables['tours'][['num_ob_stops', 'tour_purpose', 'start_tod', 'end_tod']].groupby(['tour_purpose', 'start_tod', 'end_tod']).sum().reset_index().sort_values(['end_tod', 'start_tod', 'tour_purpose'])
jstops_ib_tod = tables['unique_joint_tours'][['num_ib_stops', 'tour_purpose', 'start_tod', 'end_tod']].groupby(['tour_purpose', 'start_tod', 'end_tod']).sum().reset_index().sort_values(['end_tod', 'start_tod', 'tour_purpose'])
jstops_ob_tod = tables['unique_joint_tours'][['num_ob_stops', 'tour_purpose', 'start_tod', 'end_tod']].groupby(['tour_purpose', 'start_tod', 'end_tod']).sum().reset_index().sort_values(['end_tod', 'start_tod', 'tour_purpose'])

stops_ib_tod.to_csv(vizOutputDir + r'\todStopsIB.csv', index = False)
stops_ob_tod.to_csv(vizOutputDir + r'\todStopsOB.csv', index = False)
jstops_ib_tod.to_csv(vizOutputDir + r'\todStopsIB_joint.csv', index = False)
jstops_ob_tod.to_csv(vizOutputDir + r'\todStopsOB_joint.csv', index = False)

# prepare input for visualizer
todDepProfile_vis = todDepProfile.reset_index().rename(columns = {'start_period': 'id'})
todDepProfile_vis = pd.melt(todDepProfile_vis, ['id']).rename(columns = {'variable': 'purpose', 'value': 'freq_dep'})
todArrProfile_vis = todArrProfile.reset_index().rename(columns = {'end_period': 'id'})
todArrProfile_vis = pd.melt(todArrProfile_vis, ['id']).rename(columns = {'variable': 'purpose', 'value': 'freq_arr'})
todDurProfile_vis = todDurProfile.reset_index().rename(columns = {'tourdur': 'id'})
todDurProfile_vis = pd.melt(todDurProfile_vis, ['id']).rename(columns = {'variable': 'purpose', 'value': 'freq_dur'})

todProfile_vis = todDepProfile_vis.copy()
todProfile_vis['freq_arr'] = todArrProfile_vis['freq_arr']
todProfile_vis['freq_dur'] = todDurProfile_vis['freq_dur']
todProfile_vis = todProfile_vis.sort_values(['id', 'purpose']).fillna(0)
todProfile_vis.to_csv(vizOutputDir + r'\todProfile_vis.csv')

t9 = time.time()

print('Summarizing Tour Mode')

def SummarizeTourMode(itours, jtours, qry):
    isubset = itours[['AUTOSUFF', 'TOURMODE', 'TOURPURP']].query(qry)
    jsubset = jtours[['AUTOSUFF', 'TOURMODE', 'JOINT_PURP']].query(qry)
    itoursByMode = pd.crosstab(isubset['TOURMODE'], isubset['TOURPURP'])
    jtoursByMode = pd.crosstab(jsubset['TOURMODE'], jsubset['JOINT_PURP'])

    profile = pd.DataFrame(index = itoursByMode.index)
    profile['work'] = itoursByMode[1]
    profile['univ'] = itoursByMode[2]
    profile['sch'] = itoursByMode[3]
    profile['imain'] = itoursByMode[4] + itoursByMode[5] + itoursByMode[6]
    profile['idisc'] = itoursByMode[7] + itoursByMode[8] + itoursByMode[9]
    profile['jmain'] = jtoursByMode[5] + jtoursByMode[6]
    profile['jdisc'] = jtoursByMode[7] + jtoursByMode[8] + jtoursByMode[9]
    profile['atwork'] = itoursByMode[10]
    profile['Total'] = profile.sum(1)
    
    return profile.fillna(0)

tmodeAS0Profile = SummarizeTourMode(tables['tours'], tables['unique_joint_tours'], 'AUTOSUFF == 0')
tmodeAS1Profile = SummarizeTourMode(tables['tours'], tables['unique_joint_tours'], 'AUTOSUFF == 1')
tmodeAS2Profile = SummarizeTourMode(tables['tours'], tables['unique_joint_tours'], 'AUTOSUFF == 2')

tmodeAS0Profile.to_csv(vizOutputDir + r'\tmodeAS0Profile.csv', index = False)
tmodeAS1Profile.to_csv(vizOutputDir + r'\tmodeAS1Profile.csv', index = False)
tmodeAS2Profile.to_csv(vizOutputDir + r'\tmodeAS2Profile.csv', index = False)

tmodeAS0Profile_vis = pd.melt(tmodeAS0Profile.reset_index(), ['TOURMODE']).rename(columns = {'TOURMODE': 'id', 'variable': 'purpose', 'value': 'freq_as0'}).set_index(['id', 'purpose'])
tmodeAS1Profile_vis = pd.melt(tmodeAS1Profile.reset_index(), ['TOURMODE']).rename(columns = {'TOURMODE': 'id', 'variable': 'purpose', 'value': 'freq_as1'}).set_index(['id', 'purpose'])
tmodeAS2Profile_vis = pd.melt(tmodeAS2Profile.reset_index(), ['TOURMODE']).rename(columns = {'TOURMODE': 'id', 'variable': 'purpose', 'value': 'freq_as2'}).set_index(['id', 'purpose'])

#tmodeProfile_vis_index = pd.melt(pd.DataFrame(np.ones_like(tmodeAS0Profile), tmodeAS0Profile.index, tmodeAS0Profile.columns).reset_index(), ['TOURMODE']).set_index(['id', 'purpose']).index

#tmodeProfile_vis = tmodeAS0Profile_vis.copy()
#tmodeProfile_vis['freq_as1'] = tmodeAS1Profile_vis['freq_as1']
#tmodeProfile_vis['freq_as2'] = tmodeAS2Profile_vis['freq_as2']
#tmodeProfile_vis = tmodeProfile_vis.fillna(0)
tmodeProfile_vis = pd.DataFrame({'freq_as0': tmodeAS0Profile_vis['freq_as0'],
                                 'freq_as1': tmodeAS1Profile_vis['freq_as1'],
                                 'freq_as2': tmodeAS2Profile_vis['freq_as2']}).fillna(0).reset_index()
tmodeProfile_vis['freq_all'] = tmodeProfile_vis['freq_as0'] + tmodeProfile_vis['freq_as1'] + tmodeProfile_vis['freq_as2']
tmodeProfile_vis['id'] = tmodeProfile_vis['id'].astype(str)
tmodeProfile_vis = tmodeProfile_vis.sort_values(['purpose', 'id'])
tmodeProfile_vis.to_csv(vizOutputDir + r'\tmodeProfile_vis.csv', index = False) #May need to fix order

t10 = time.time()

print('Summarizing Non-Mandatory Tour Lengths')

bins = list(range(41)) + [9999]
tourDistProfile = pd.DataFrame(index = bins[1:-1] + [41])
tourDistProfile['esco'] = np.histogram(tables['tours'][['TOURPURP', 'tour_distance']].query('TOURPURP == 4')['tour_distance'], bins)[0]
tourDistProfile['imain'] = np.histogram(tables['tours'][['TOURPURP', 'tour_distance']].query('TOURPURP >= 5 and TOURPURP <= 6')['tour_distance'], bins)[0]
tourDistProfile['idisc'] = np.histogram(tables['tours'][['TOURPURP', 'tour_distance']].query('TOURPURP >= 7 and TOURPURP <= 9')['tour_distance'], bins)[0]
tourDistProfile['jmain'] = np.histogram(tables['unique_joint_tours'][['JOINT_PURP', 'tour_distance']].query('JOINT_PURP >= 5 and JOINT_PURP <= 6')['tour_distance'], bins)[0]
tourDistProfile['jdisc'] = np.histogram(tables['unique_joint_tours'][['JOINT_PURP', 'tour_distance']].query('JOINT_PURP >= 7 and JOINT_PURP <= 9')['tour_distance'], bins)[0]
tourDistProfile['atwork'] = np.histogram(tables['tours'][['TOURPURP', 'tour_distance']].query('TOURPURP == 10')['tour_distance'], bins)[0]

tourDistProfile.to_csv(vizOutputDir + r'\nonMandTourDistProfile.csv')

tourDistProfile_vis = tourDistProfile.copy()
tourDistProfile_vis['Total'] = tourDistProfile_vis.sum(1)
tourDistProfile_vis = pd.melt(tourDistProfile_vis.reset_index(), ['index']).rename(columns = {'index': 'distbin', 'variable': 'PURPOSE', 'value': 'freq'})
tourDistProfile_vis.to_csv(vizOutputDir + r'\tourDistProfile_vis.csv')

t11 = time.time()

print('Done')
print(t11 - t0)