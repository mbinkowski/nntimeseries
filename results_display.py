from __init__ import *
from keras.models import load_model

key = 'artificial'
mindate = '2017-02-13'
files = [f for f in os.listdir('results') if (key in f) and ('.pkl' in f)]
read_tables = []
for f in files:
    try:
        df = pd.read_pickle('results/' + f)
        df = df[df['dt'] > pd.Timestamp(mindate)]
        df['epochs'] = df['loss'].apply(lambda x: len(x))
        df = df[df['epochs'] > 5]
        if df.shape[0] == 0:
            continue
        if 'objective' not in df.columns:
            df['objective'] = 'regr' if ('regr' in f) else 'class'
        if 'returns' not in df.columns:
            df['returns'] = ('ret' in f)
        elif 'both' in f:
            df = df[df['returns']] ###### screwed up classification loss in 'both'
        if 'full_shift' not in df.columns:
            df['full_shift'] = True
        df['NeX'] = ('NeX' in f)
        if 'batch_size' not in df.columns:
            df['batch_size'] = 64
        if 'channels' not in df.columns:
            if 'filters' in df.columns:
                df['channels'] = df['filters']
            else:
                df['channels'] = 1
        else:
            df.loc[np.isnan(df['channels']), 'channels'] = 1
        if 'model' in df.columns:
            pass
        elif ('benchmark' in f) or ('lr' in f):
            df['model'] = 'lr'
        elif 'cnn' in f:
            df['model'] = 'cnn'
        elif 'gru' in f:
            df['model'] = 'gru'
        elif 'lstm' in f:
            df['model'] = 'lstm'
        elif 'value_importance3' in f:
            df['model'] = 'so3'
        elif ('cvi2' in f) or ('cnnvi2' in f):
            df['model'] = 'cvi2'
        elif ('cvi' in f) or ('cnnvi' in f):
            df['model'] = 'cvi'            
        else:
            df['model'] = 'so'
        keys, funcs, names = [], [], []
        df['best_mse'] = np.NaN
        df['best_acc'] = np.NaN
        if ('class' in f) or all(df['objective'] == 'class'):
            if 'val_main_output_acc' in df.columns:
                keys.append('val_main_output_acc')
                funcs.append([np.nanmax, np.nanargmax])
                names.append('best_acc')
            elif 'val_acc' in df.columns:
                keys.append('val_acc')
                funcs.append([np.nanmax, np.nanargmax])
                names.append('best_acc')
        if ('regr' in f) or all(df['objective'] == 'regr') :
            if 'val_main_output_loss' in df.columns:
                keys.append('val_main_output_loss')
                funcs.append([np.nanmin, np.nanargmin])
                names.append('best_mse')
            elif 'val_loss' in df.columns:
                keys.append('val_loss')
                funcs.append([np.nanmin, np.nanargmin])        
                names.append('best_mse')
        for k, func, n in zip(keys, funcs, names):
            df[n] = df[k].apply(lambda x: func[0](x))
            df['best_epoch'] = df[k].apply(lambda x: func[1](x))
        df['file'] = ''.join(f.split(key)) if (len(key) > 0) else f
        df['time_per_epoch'] = df['training_time'] / df['epochs']
        df['time_to_best'] = df['best_epoch'] * df['time_per_epoch']
        read_tables.append(df.copy())
    except Exception as e:
        print(e)
        
def total_params(model):
    return np.sum([np.sum([np.prod(K.eval(w).shape) for w in l.trainable_weights]) for l in model.layers])

all_results = pd.concat(read_tables)  
all_results.reset_index(inplace=True)
# if 'total_params' not in all_results.columns:
#     all_results['total_params'] = all_results['hdf5'].apply(lambda x: total_params(load_model(x)))
# else:
#     idx = np.isnan(all_results['total_params'])
#     all_results.loc[idx, 'total_params'] = all_results.loc[idx, 'hdf5'].apply(lambda x: total_params(load_model(x)))
all_results.to_pickle(WDIR + '/results/all_results.pkl')                

def get_pivot(setting, df, valcols=[]):
    if setting['objective'] == 'class':
        best = 'best_acc'
        func = max
    else:
        best = 'best_mse'
        func = min
    df0 = df.copy()
    for k, v in setting.items():
        df0 = df0[df0[k] == v]
    df0 = df0[df0.groupby(['data', 'model'])[best].transform(func) == df0[best]]
    pivs = [df0.pivot(index='data', columns='model', values=best)]
    for col in valcols:
        piv = df0.pivot(index='data', columns='model', values=col)
        piv.rename(columns=dict([(c, col[:3] + '_' + c) for c in piv.columns]), inplace=True)
        pivs.append(piv.copy())
    return df0, pd.concat(pivs, axis=1)

print(all_results.groupby('file').count())

setting = dict(
    objective = 'regr',
    train_share = (.8, 1.),
    diffs = False,
#     architecture = {'lambda': True, 'softmax':False, 'nonneg':False}
)

df0, pivot = get_pivot(setting, all_results, ['total_params', 'time_to_best'])
print(pivot)


