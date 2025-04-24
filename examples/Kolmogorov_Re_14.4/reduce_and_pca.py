import numpy as np
from matplotlib import pyplot as plt
import h5py
from sklearn.decomposition import PCA
from derivatives import kolmo_stuff

from dsyre4py import symmred
from dsyre4py.tools import prod

#Define the symmetry operators:

def rotate(u):
    u_fft = np.fft.rfft2(u.copy(),axes=(1,2))
    u_out_fft = np.conjugate(u_fft)
    return(np.fft.irfft2(u_out_fft,axes=(1,2))
)

def shift_reflect(u):
    u_fft = np.fft.rfft(u.copy(),axis=(2))
    #ky = np.arange(0,u_fft.shape[2])
    ky = np.arange(0,int(u_fft.shape[2]))
    times = -1* np.pi/2

    ky_expanded = ky 
    times_expanded = times
    for i in range(u_fft.ndim):
        if i != 2:
            ky_expanded = np.expand_dims(ky_expanded,axis=i)
        times_expanded = np.expand_dims(times_expanded,axis=i)
    expo = 1j * ky_expanded * times_expanded
    u_out_fft = -1 * u_fft * np.exp(expo)
    u_out = np.fft.irfft(u_out_fft,axis=(2))
    u_out_fft = np.fft.fft(u_out,axis=(1))
    u_out_fft[:,0,:] = u_out_fft[:,0,:]
    u_out_fft[:,1:,:] = np.flip(u_out_fft[:,1:,:],axis=1)
    return np.real(np.fft.ifft(u_out_fft,axis=(1)))


def translate(u,s=0):
    u_fft = np.fft.rfft(u.copy(),axis=(1))
    #kx = np.concatenate([np.arange(-int(u_fft.shape[1]/2),0),np.arange(0,int(u_fft.shape[1]/2))],axis=0)
    kx = np.arange(0,int(u_fft.shape[1]))
    times = -s
    kx_expanded = kx
    times_expanded = times
    for i in range(u_fft.ndim):
        if i != 1:
            kx_expanded = np.expand_dims(kx_expanded,axis=i)
    for i in range(u_fft.ndim-times.ndim+1):
        if i != 0:
            times_expanded = np.expand_dims(times_expanded,axis=i)
    expo = 1j * kx_expanded * times_expanded
    u_out_fft = u_fft * np.exp(expo)
    return np.fft.irfft(u_out_fft,axis=(1))

###

dt = 0.1
Re = 14.4
weight_path = 'weights_{}_kolmo'.format(Re)

try:
    fnstr="data/Kolmogorov_Re_{}_dt_{}.h5".format(Re,dt)
    hf = h5py.File(fnstr, 'r')
except:
    print('Data not downloaded.')
    user_input = input("Do you want to download it now? (y/n): ")
    if user_input.lower() == "y":
        import urllib.request
        import shutil     
        URL = "https://zenodo.org/records/15195938/files/kolmogorov_Re_14.4_64x64_dt_0.1.h5?download=1"

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



data = np.array(hf.get('vorticity')[500:2000,:,:])
hf.close()

kom = kolmo_stuff(data,Re=Re)
no_used = 50
#define trajectory length for dynamic inversion 
length_traj  = int(10/dt)
max_modes = 10
modes_step = 1

test_train_ratio = 1/5
ind_split = int(data.shape[0]*(1-test_train_ratio))

T = np.linspace(0,length_traj*dt,length_traj)

sr = symmred([translate],[1],[shift_reflect,rotate],[4,2],data.shape,no_used=no_used, weight_path=weight_path,RHS=kom.get_deriv_ts,spatial_derivs=[kom.get_deriv_x_ts],dt=dt)


fig_proj,ax_proj = plt.subplots(1,2,figsize=(6,3))   
fig_proj.suptitle('PCA Projections', fontsize=16) 
ax_proj[0].set_title('No Symm Red')
ax_proj[1].set_title('Symm Red')

fig_eigs,ax_eigs = plt.subplots(1,1,figsize=(3,3))   

train_data = data.copy()[:ind_split]
test_data = data.copy()[ind_split:]

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
ax_proj[0].plot(train_pca[:,0],train_pca[:,1],'c.',ms=0.5,label='Train Data')
ax_proj[0].plot(test_pca[:,0],test_pca[:,1],'m.',ms=0.5,label='Test Data')
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

#train_data,test_data = np.split(data.copy(),2,axis=0)
train_data = data.copy()[:ind_split]
test_data = data.copy()[ind_split:]

#train_data = data.copy()
#test_data = data.copy()
print(train_data.shape,test_data.shape)

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
fig_eigs.suptitle('PCA Eigenvalues', fontsize=16) 
fig_eigs.tight_layout()
fig_eigs.savefig('fig/kolmo_chaos_eigs.pdf')

train_pca = pca.transform(reduced_train_stack)
test_pca = pca.transform(reduced_test_stack)
ax_proj[1].plot(train_pca[:,0],train_pca[:,1],'c.',ms=0.5)
ax_proj[1].plot(test_pca[:,0],test_pca[:,1],'m.',ms=0.5)
ax_proj[1].set_xlabel(r'$\chi_1$')
ax_proj[1].set_ylabel(r'$\chi_2$')
fig_proj.tight_layout()
fig_proj.savefig('fig/kolmo_chaos_proj.pdf')

errs_symmred_train = []
errs_symmred_train_dyn = []

errs_symmred_test = []
errs_symmred_test_dyn = []

fig_ts_err,ax_ts_err = plt.subplots(1,1)      
cmap=plt.cm.get_cmap(plt.cm.viridis)

train_data_split = np.split(train_data,int(train_data.shape[0]/length_traj),axis=0)
test_data_split = np.split(test_data,int(test_data.shape[0]/length_traj),axis=0)


for ind,i in enumerate(n):
    print(i,'modes')
    train_pca[:,i:] = 0.0
    test_pca[:,i:] = 0.0
    
    train_pca_real_inv = pca.inverse_transform(train_pca)
    test_pca_real_inv = pca.inverse_transform(test_pca)


    train_pca_real_inv = train_pca_real_inv.reshape(reduced_train_shape)
    test_pca_real_inv = test_pca_real_inv.reshape(reduced_test_shape)

    train_pca_inv = train_pca_real_inv[...,0] + train_pca_real_inv[...,1] * 1j
    test_pca_inv = test_pca_real_inv[...,0] + test_pca_real_inv[...,1] * 1j

    print('inverting')

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
fig_ts_err.savefig('fig/kolmo_chaos_ts_error.pdf')


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
figq.savefig('fig/kolmo_chaos_error.pdf')
plt.show()

