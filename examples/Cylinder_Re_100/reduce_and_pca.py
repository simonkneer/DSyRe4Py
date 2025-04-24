import numpy as np
from matplotlib import pyplot as plt
import h5py
from sklearn.decomposition import PCA
from dsyre4py import symmred
from dsyre4py.tools import prod

#Define the symmetry operators:

def reflect(u):
    u_out = np.flip(u.copy(),axis=2)
    return(-1 * u_out)

###

dt = 0.02
Re = 100
weight_path = 'weights_{}_cylinder'.format(Re)



try:
    fnstr="data/cylinder_Re_{}_450x200_dt_{}.h5".format(Re,dt)
    hf = h5py.File(fnstr, 'r')
except:
    print('Data not downloaded.')
    user_input = input("Do you want to download it now? (y/n): ")
    if user_input.lower() == "y":
        import urllib.request
        import shutil
        URL = "https://zenodo.org/records/15195938/files/cylinder_Re_100_450x200_dt_0.02.h5?download=1"

        with urllib.request.urlopen(URL) as response, open(fnstr, 'wb') as out_file:
            try:
                length = response.getheader('content-length')
                print('File has size of approx {} GB. This might take a while ...'.format(length*1E-9))
            except:
                print('')
            shutil.copyfileobj(response, out_file)        
            print('Download completed.')        
        hf = h5py.File(fnstr, 'r')            
    else:
        print("Exiting...")    
        sys.exit()

data = np.array(hf.get('VORT')[:1000])
hf.close()

#define trajectory length for dynamic inversion 
length_traj  = 25#int(250/dt)
no_used = 20
max_modes = 10
modes_step = 1

T = np.linspace(0,length_traj*dt,length_traj)

sr = symmred([],[],[reflect],[2],data.shape,weight_path=weight_path,no_used=no_used)


fig_proj,ax_proj = plt.subplots(1,2,figsize=(6,3))   
fig_proj.suptitle('PCA Projections', fontsize=16) 
ax_proj[0].set_title('No Symm Red')
ax_proj[1].set_title('Symm Red')

fig_eigs,ax_eigs = plt.subplots(1,1,figsize=(3,3))   
fig_eigs.suptitle('PCA Eigenvalues', fontsize=16) 

train_data,test_data = np.split(data.copy(),2,axis=0)
#train_data = data.copy()
#test_data = data.copy()
train_data_stack = train_data.reshape((train_data.shape[:1]) + (prod(train_data.shape[1:]),))
test_data_stack = test_data.reshape((test_data.shape[:1]) + (prod(test_data.shape[1:]),))

pca = PCA(n_components=max_modes)
pca.fit(train_data_stack)
train_pca = pca.transform(train_data_stack)
test_pca = pca.transform(test_data_stack)

eigs = pca.explained_variance_ratio_
ax_eigs.plot(eigs,'b.',label='No Symm Red')
ax_proj[0].plot(train_pca[:,0],train_pca[:,1],'c.',label='Train Data')
ax_proj[0].plot(test_pca[:,0],test_pca[:,1],'m.',label='Test Data')
ax_proj[0].legend(loc='lower left')
ax_proj[0].set_xlabel(r'$\chi_1$')
ax_proj[0].set_ylabel(r'$\chi_2$')

errs_train = []
errs_test = []
n = np.arange(max_modes,0,-modes_step,dtype=int)

for i in n:
    test_pca[:,i:] = 0.0
    train_pca[:,i:] = 0.0

    test_pca_inv = pca.inverse_transform(test_pca)
    errs_test.append(np.sqrt(np.mean((test_pca_inv-test_data_stack)**2)))
    train_pca_inv = pca.inverse_transform(train_pca)
    errs_train.append(np.sqrt(np.mean((train_pca_inv-train_data_stack)**2)))    

train_data,test_data = np.split(data.copy(),2,axis=0)
#train_data = data.copy()
#test_data = data.copy()

print('reducing ...')
reduced_train = sr.reduce_all(train_data)
reduced_test = sr.reduce_all(test_data)

reduced_train_real = np.stack([np.real(reduced_train),np.imag(reduced_train)],axis=-1)
reduced_test_real = np.stack([np.real(reduced_test),np.imag(reduced_test)],axis=-1)

reduced_train_shape = reduced_train_real.shape
reduced_test_shape = reduced_test_real.shape # 

reduced_train_stack = reduced_train_real.reshape((reduced_train_real.shape[:1]) + (prod(reduced_train_real.shape[1:]),))
reduced_test_stack = reduced_test_real.reshape((reduced_test_real.shape[:1]) + (prod(reduced_test_real.shape[1:]),))

print('pcaing ...')

pca = PCA(n_components=max_modes)
pca.fit(reduced_train_stack)
eigs = pca.explained_variance_ratio_
ax_eigs.plot(eigs,'r.',label='Symm Red') 
ax_eigs.legend(loc='lower left')
ax_eigs.set_yscale('log')
ax_eigs.set_xlabel('$i$')
ax_eigs.set_ylabel('$\lambda_i$')
fig_eigs.tight_layout()
fig_eigs.savefig('fig/cylinder_period_eigs.pdf')

train_pca = pca.transform(reduced_train_stack)
test_pca = pca.transform(reduced_test_stack)
ax_proj[1].plot(train_pca[:,0],train_pca[:,1],'c.')
ax_proj[1].plot(test_pca[:,0],test_pca[:,1],'m.')
ax_proj[1].set_xlabel(r'$\chi_1$')
ax_proj[1].set_ylabel(r'$\chi_2$')
fig_proj.tight_layout()
fig_proj.savefig('fig/cylinder_period_proj.pdf')

errs_symmred_train = []
errs_symmred_train_dyn = []

errs_symmred_test = []
errs_symmred_test_dyn = []

fig_ts_err,ax_ts_err = plt.subplots(1,1)      
cmap=plt.cm.get_cmap(plt.cm.viridis)

train_data_split = np.split(train_data,int(train_data.shape[0]/length_traj),axis=0)
test_data_split = np.split(test_data,int(test_data.shape[0]/length_traj),axis=0)


for ind,i in enumerate(n):
    train_pca[:,i:] = 0.0
    test_pca[:,i:] = 0.0
    
    train_pca_real_inv = pca.inverse_transform(train_pca)
    test_pca_real_inv = pca.inverse_transform(test_pca)


    train_pca_real_inv = train_pca_real_inv.reshape(reduced_train_shape)
    test_pca_real_inv = test_pca_real_inv.reshape(reduced_test_shape)

    train_pca_inv = train_pca_real_inv[...,0] + train_pca_real_inv[...,1] * 1j
    test_pca_inv = test_pca_real_inv[...,0] + test_pca_real_inv[...,1] * 1j

    train_pca_inv_split = np.split(train_pca_inv,int(train_pca_inv.shape[0]/length_traj),axis=0)
    test_pca_inv_split = np.split(test_pca_inv,int(test_pca_inv.shape[0]/length_traj),axis=0)

    temp_err_symmred_train = 0
    temp_err_symmred_train_dyn = 0

    ts_err_list_dyn = []
    ts_err_list_stat = []
    for inv_split,data_comp in zip(train_pca_inv_split,train_data_split):
        split_dyn = np.real(sr.inv_reduce_all_dynamic(inv_split,data_comp))    
        split_static, data_comp_static = sr.inv_reduce_all_static(inv_split,data_comp)
        split_static = np.real(split_static)
        data_comp_static = np.real(data_comp_static)

        temp_err_symmred_train_dyn += np.mean((split_dyn-data_comp)**2)
        temp_err_symmred_train += np.mean((split_static-data_comp_static)**2)
        ts_err_list_dyn.append(np.sqrt(np.mean((split_dyn-data_comp)**2,axis=(1,2))))
        ax_ts_err.plot(T,ts_err_list_dyn[-1],c=cmap(ind/n.shape[0]*0.5),lw=0.1)

    ax_ts_err.plot(T,np.sqrt(np.mean(np.stack(ts_err_list_dyn,axis=0)**2,axis=0)),'--',c=cmap(ind/n.shape[0]*0.5),label='Train {} Modes'.format(i))

    errs_symmred_train_dyn.append(np.sqrt(temp_err_symmred_train_dyn/len(train_pca_inv_split)))
    errs_symmred_train.append(np.sqrt(temp_err_symmred_train/len(train_pca_inv_split)))

    temp_err_symmred_test = 0
    temp_err_symmred_test_dyn = 0    
    
    ts_err_list_dyn = []
    ts_err_list_stat = []
    for inv_split,data_comp in zip(test_pca_inv_split,test_data_split):
        split_dyn = np.real(sr.inv_reduce_all_dynamic(inv_split,data_comp))    
        split_static, data_comp_static = sr.inv_reduce_all_static(inv_split,data_comp)
        split_static = np.real(split_static)
        data_comp_static = np.real(data_comp_static)

        temp_err_symmred_test_dyn += np.mean((split_dyn-data_comp)**2)
        temp_err_symmred_test += np.mean((split_static-data_comp_static)**2)
        ts_err_list_dyn.append(np.sqrt(np.mean((split_dyn-data_comp)**2,axis=(1,2))))
        ax_ts_err.plot(T,ts_err_list_dyn[-1],c=cmap(ind/n.shape[0]*0.5),lw=0.1)

    ax_ts_err.plot(T,np.sqrt(np.mean(np.stack(ts_err_list_dyn,axis=0)**2,axis=0)),c=cmap(ind/n.shape[0]*0.5),label='Test {} Modes'.format(i))

    errs_symmred_test_dyn.append(np.sqrt(temp_err_symmred_test_dyn/len(test_pca_inv_split)))
    errs_symmred_test.append(np.sqrt(temp_err_symmred_test/len(test_pca_inv_split)))    


#print('symmred real',np.mean(np.abs(inv_reduced_cont-testo_cont)))
ax_ts_err.set_yscale('log') 
ax_ts_err.legend(loc='upper left', bbox_to_anchor=(1.0, 1., ))
ax_ts_err.set_xlabel('$t$')
ax_ts_err.set_ylabel(r'$\|\omega-\hat{\omega}\|$')
ax_ts_err.set_xlim([T[0],T[-1]])
fig_ts_err.suptitle('Timeseries Reconstruction Errors', fontsize=16) 
fig_ts_err.tight_layout()
fig_ts_err.savefig('fig/cylinder_period_ts_error.pdf')


figq,axq = plt.subplots(1,1,figsize=(6,3))     
figq.suptitle('Reconstruction Errors', fontsize=16) 
axq.plot(n,errs_train,'b>',label='Train, No Symm Red')
axq.plot(n,errs_test,'b<',label='Test, No Symm Red')

axq.plot(n,errs_symmred_train,'r>',label='Train, Symm Red Stat')
axq.plot(n,errs_symmred_test,'r<',label='Test, Symm Red Stat')

axq.plot(n,errs_symmred_train_dyn,'g>',label='Train, Symm Red Dyn')
axq.plot(n,errs_symmred_test_dyn,'g<',label='Test, Symm red Dyn')

axq.set_yscale('log')
axq.legend(loc='lower left')
axq.set_xlabel('$n_{modes}$')
axq.set_ylabel(r'$\|\omega-\hat{\omega}\|$')
figq.tight_layout()
figq.savefig('fig/cylinder_period_error.pdf')
plt.show()

