import numpy as np
from .tools import optimize
from .tools import prod
import pickle as pk
import os
from scipy import integrate

class symmred:

    def __init__(self,cont_operator_list,axes, disc_operator_list, cyclicities,field_size,*,weight_path='weight',no_used=20,RHS = None, spatial_derivs = None, dt = 0.1):
        """Initializes a symmetry reduction class

        Args:
            cont_operator_list (list): List of functions that represent the action of the continuous symmetries
            axes (list): List of axes on which the continuous symmetry acts
            k_cont (list): List containing which wavenumber to use for continuous symmetry reduction
            disc_operator_list (list): List of functions that represent the action of the discrete symmetries
            cyclicities (list): List of integers matching the cylicities of the operators in disc_operator_list
            field_size (list): Shape of the field to be reduced
            weight_path (str, optional): Absolute/Relative path to the symmetry reducing weights. Defaults to 'weight'.
            no_used (int, optional): Number of N largest (on average) grid points to use for reducing. Defaults to 20.
            RHS (optional): Function that gives the time derivative of the input field. Needed for reconstruction eq.
            spatial_derivs (optional): List of functions that give the spatial derivatives of the input field. Needed for reconstruction eq.        
        """
        self.n = cyclicities
        self.disc_op = disc_operator_list
        self.RHS = RHS
        self.spatial_derivs = spatial_derivs
        self.dt = dt
        self.cont_op = cont_operator_list
        self.axes = axes
        self.no_used = no_used
        self.orig_dims = len(field_size)
        self.total_op_disc = len(disc_operator_list)
        self.total_op_cont = len(cont_operator_list)
        self.weight_path = weight_path
        self.phase_list = []
        #Try loading the weights, if not initialize
        try:
            with open(self.weight_path, 'rb') as f:
                self.cont_weights,self.disc_weights  = pk.load(f)
        except:
            print('path does not exist - optimizing new weights')
            self.disc_weights = []
            self.cont_weights = []

        #Calculate principal roots of unity for the discrete symmetries
        self.root = []
        for n in self.n:
            self.root.append(np.exp(1j*(2*np.pi)/n))

    def gen_copies(self,field,op_num):
        """Generates all symmetry copies for symmetries with index \leq op_num
            if for example the list of cyclicities is [4,3,2] and we call this function with
            op_num = 1, this would return 4\times 3 symmetry copies corresponding to the combinations of 
            symmetry 0 and 1.

        Args:
            field (array): Field of which symmetry copies are generated
            op_num (int): highest symmetry that is considered. Generates symmetry copies for all lower ones as well

        Returns:
            array: Symmetry copies of field. Copies are stacked along last axis.
        """
        out = self.operator_field(field,op_num)
        if(op_num!=0):
            out = self.gen_copies(out,op_num-1)
        return out

    def operator_field(self,field,op_num):
        """This returns an array of fields of size n where each entry i
        corresponds to the operator being applied i times

        Args:
            field (array): Input field
            op_num (int): Operator index

        Returns:
            array: Operator applied to field
        """
        operator_field = [field]
        for i in range(self.n[op_num]-1):
            operator_field.append(self.disc_op[op_num](operator_field[-1]))
        return np.stack(operator_field,axis=-1)

    def bases(self,field,op_num):
        """This function calculates the necessary bases for forming the symmetry-invariant 
        polynomials. Since these can *not* be gained by simply applying the operator, but rather
        we need to apply the operator *before* we reduce any previous symmetries, we call 
        the symmetry reduction before

        Args:
            field (array): input field
            op_num (int): operator index

        Returns:
            array: bases for the symmetry reducing polynomials
        """
        if(op_num==0):
            operator_field  = self.cont_reduce(self.operator_field(field,op_num))
        else:
            operator_field = self.disc_reduce(self.operator_field(field,op_num),op_num-1)
        bases = []
        for i in np.arange(1,self.n[op_num]+1):
            temp = np.take(operator_field,0,axis=-(1+op_num))
            bases.append(temp.astype(complex)/self.n[op_num])
            for j in np.arange(1,self.n[op_num]):
                temp = np.take(operator_field,j,axis=-(1+op_num))
                bases[-1] += self.root[op_num]**(-i*j)*temp/self.n[op_num]
        return np.stack(bases,axis=-(1+op_num))
    
    def get_reducing_num(self,base,op_num,tot_weight):
        """Calculates the symmetry reducing unit number from the
        first operator basis set (passed) 

        Args:
            base (array): First operator basis
            op_num (int): operator index
            tot_weight (array): weights

        Returns:
            array: symmetry reducing unit number
        """
        
        if(len(base.shape)!=len(tot_weight.shape)):
            #padding is always for following operations, which are placed in front
            axis_match = tuple(range(self.orig_dims,self.orig_dims+len(base.shape)-len(tot_weight.shape)))
            tot_weight = np.expand_dims(tot_weight,axis=axis_match)
        if op_num!=0:
            axes = tuple(list(range(1, self.orig_dims))+list(range(-op_num, 0)))
        else:
            axes = tuple(range(1, self.orig_dims))
        #print(base.shape,tot_weight.shape,op_num)
        reducing_num = np.sum(tot_weight*base,axis=axes)
        
        reducing_num = np.expand_dims(reducing_num,axis=axes)
        reducing_num = reducing_num/np.abs(reducing_num)
        return reducing_num

    def disc_reduce(self,field,op_num):
        """Reduces the continuous symmetries up to operator op_num and all continuous symmetries

        Args:
            field (array): input field
            op_num (int): operator index

        Returns:
            array: discrete symmetry reduced fields            
        """
        reduced_field = []
        bases = self.bases(field,op_num)
        try:
            tot_weight = self.disc_weights[op_num].copy()
        except:
            print('No weights found' +'\n'+ 'making new weights')
            shape = np.take(bases,0,axis=-(1+op_num)).shape
            if(op_num == 0):
                lst = list(shape)[:self.orig_dims]
            else:
                lst = list(shape)[:self.orig_dims] + list(shape)[-op_num:]
            lst[0] = 1
            shape = tuple(lst)
            weight_re = np.zeros(shape)
            weight_im = np.zeros(shape)
            weight_re_stacked = weight_re.reshape(prod(shape))
            weight_im_stacked = weight_im.reshape(prod(shape))
            field_for_optim = np.take(bases,0,axis=-(1+op_num))
            if op_num!=0:
                axes = tuple(list(range(0, self.orig_dims))+list(range(-op_num, 0)))
            else:
                axes = tuple(range(0, self.orig_dims))
            axes = tuple(axis if axis >= 0 else axis + field_for_optim.ndim for axis in axes)
            slices = tuple(0 if i not in axes else slice(None) for i in range(field_for_optim.ndim))
            field_for_optim = field_for_optim[slices][:]
            field_for_optim = field_for_optim.reshape((field_for_optim.shape[:1]) + (prod(field_for_optim.shape[1:]),))
            norm_field_form_optim = np.mean(field_for_optim**2,axis=0)
            ind = np.argpartition(-norm_field_form_optim, self.no_used)[:self.no_used]
            field_for_optim = field_for_optim [:,ind]
            weight = optimize(field_for_optim)
            weight_re_stacked[ind], weight_im_stacked[ind] = weight#np.split(weight_stacked,2)
            weight_re = weight_re_stacked.reshape(shape)
            weight_im = weight_im_stacked.reshape(shape)
            tot_weight = weight_re + 1j * weight_im  
            self.disc_weights.append(tot_weight)

        reducing_num = self.get_reducing_num(np.take(bases,0,axis=-(1+op_num)),op_num,tot_weight)

        for ind in range(bases.shape[-(1+op_num)]):
            base = np.take(bases,ind,axis=-(1+op_num))
            if ind == self.n[op_num]-1:
                reduced_field.append(base)
            else:
                reduced_field.append(base*reducing_num**(self.n[op_num]-ind-1))
        reduced_field = np.stack(reduced_field,axis=-(1+op_num))
        return(reduced_field)

    def cont_reduce(self,field,save_phase = True):
        """This function performs continuous symmetry reduction for all continuous symmetries

        Args:
            field (array): original input field
            save_phase (bool, optional): Flag whether the phase should be saved in self.phase_list. Defaults to True.

        Returns:
            array: discrete symmetry reduced fields                
        """
        reduced_field = field
        for op_num in range(self.total_op_cont):
            field_fft = np.fft.fft(reduced_field,axis=self.axes[op_num])
            field_taken = np.take(field_fft,1,axis=self.axes[op_num])
            if self.orig_dims == 2:
                #if only one spatial dimension exists we cannot optimize weights
                num = field_taken
            else:
                try:
                    weight = self.cont_weights[op_num].copy()
                except:            
                    field_for_optim = field_taken.copy()

                    shape = field_for_optim.shape

                    lst = list(shape)[0:self.orig_dims-1]
                    lst[0] = 1
                    shape = tuple(lst)      
                    weight_re = np.zeros(shape)                
                    weight_re_stacked = weight_re.reshape(prod(shape))
                    axes = tuple(range(0, self.orig_dims-1))
                    axes = tuple(axis if axis >= 0 else axis + field_for_optim.ndim for axis in axes)
                    slices = tuple(0 if i not in axes else slice(None) for i in range(field_for_optim.ndim))
                    field_for_optim = field_for_optim[slices][:]
                    field_for_optim = field_for_optim.reshape((field_for_optim.shape[:1]) + (prod(field_for_optim.shape[1:]),))
                    norm_field_form_optim = np.mean(field_for_optim**2,axis=0)
                    ind = np.argpartition(-norm_field_form_optim, self.no_used)[:self.no_used]
                    field_for_optim = field_for_optim [:,ind]
                    weight_re_stacked[ind] = optimize(field_for_optim,complex_val=False)
                    weight_re = weight_re_stacked.reshape(shape)
                    
                    weight = weight_re
                    self.cont_weights.append(weight.copy())

                if(len(field_taken.shape)!=len(weight.shape)):
                    #padding is always for following operations, which are placed in front
                    axis_match = tuple(range(self.orig_dims-1,self.orig_dims+len(field_taken.shape)-len(weight.shape)-1))
                    weight = np.expand_dims(weight,axis=axis_match)

                sum_axes = tuple(range(1,self.orig_dims-1))
                num = np.sum(weight*field_taken,axis=sum_axes)
            phase = np.arctan2(-np.real(num),-np.imag(num))
            if save_phase:
                self.phase_list.append(phase)
            reduced_field = self.cont_op[op_num](reduced_field,-phase)
        return(reduced_field)
            
    def reduce_all(self,field):
        """This simply calls disc_reduce with the maximum op_num to reduce all symmetries. Also saves the optimized weights

        Args:
            field (array): input field

        Returns:
            array: fully symmetry reduced data
        """
        out = self.disc_reduce(field,self.total_op_disc-1)
        if (not os.path.isfile(self.weight_path)):
            with open(self.weight_path, 'wb') as f:
                pk.dump([self.cont_weights, self.disc_weights], f)
        return out 

                
    def inv_reduce_all_static(self,bases,field):
        """This inverts all discrete symmetry operations. For each sample the returned correct symmetry branch is simply the one with the lowest error when compared to 
        the continuous symmetry reduced field. this routine is not suitable for dynamical modelling since it just picks the minimum error at each step and hence requires information of the original field

        Args:
            bases (array): symmetry reduced field
            field (array): original input field

        Returns:
        array: bases with discrete symmetry reduction inverted and branches picked
        array: original input field but continuous symmetries removed. As a comparison for the inverted fields
        """
        #
        bases = self.inv_reduce_all_disc(bases)
        bases = np.real(self.gen_copies(bases,self.total_op_disc-1))
        bases = bases.reshape(bases.shape[:self.orig_dims] + (prod(bases.shape[self.orig_dims:]),))
        #applying copies can kick you out of the slice
        bases = self.cont_reduce(bases,save_phase = False)

        field_cont = self.cont_reduce(field,save_phase = False)

        sum_axes = tuple(range(1,self.orig_dims))
        inds = np.argmin(np.sum((bases-np.expand_dims(field_cont,axis=-1))**2,axis=sum_axes),axis=-1)
        rows = np.arange(inds.shape[0])
        bases_picked = bases[rows,...,inds]#np.take(bases,inds,axis=-1)
     
        return(bases_picked,field_cont)

    def inv_reduce_all_dynamic(self,bases,field):
        """This routine inverts the symmetry reduction by considering continuity of the derivative of the time series.
        The correct symmetry copies arising from the discrete symmetry reduction are picked by maximizing the likeness of the temporal 
        derivative of one sample to the previous sample. 
        The continuous phase is recovered by integrating the reconstruction equation: C. W. Rowley and J. E. Marsden, “Reconstruction equations and the karhunen–lo`eve expansion for systems with symmetry,” Physica D 142, 1–19 (2000).
        
        this routine is suitable for dynamical modelling since it the minimum error at the first step only and then picks succesive copies by making the derivative maximally continuous

        Args:
            bases (array): symmetry reduced field
            field (array): original input field

        Returns:
            array: symmetry reduced field with all symmetry reduction methods inverted
        """
        bases = self.inv_reduce_all_disc(bases)
        bases = np.real(self.gen_copies(bases,self.total_op_disc-1))

        #applying copies can kick you out of the slice
        self.phase_list = []    
        bases = self.cont_reduce(bases,save_phase = False)
        bases = bases.reshape(bases.shape[:self.orig_dims] + (prod(bases.shape[self.orig_dims:]),))

        #for calculating a gradient we need two points
        field_cont = self.cont_reduce(field[:2],save_phase = True)

        sum_axes = tuple(range(1,self.orig_dims))
        inds = np.argmin(np.sum((bases[:2]-np.expand_dims(field_cont,axis=-1))**2,axis=sum_axes),axis=-1)
        rows = np.arange(inds.shape[0])
        init_base = bases[rows,...,inds]

        bases_picked = []
        alphas = []
        for i in range(2):
            bases_picked.append(init_base[i])
        deriv = bases_picked[-1] - bases_picked[-2]
        sum_axes = tuple(range(0,self.orig_dims-1))
        for i in np.arange(1,bases.shape[0]-1):
            deriv_next = bases[i+1] - np.expand_dims(bases_picked[i],axis=-1)
            proj = np.sum(np.expand_dims(deriv,axis=-1)*deriv_next,axis=sum_axes)
            norm = np.expand_dims(np.sqrt(np.sum(deriv**2)),axis=-1) * np.sqrt(np.sum(deriv_next**2,axis=sum_axes))
            alphas.append(proj/norm)
            ind = np.argmax(alphas[-1])
            bases_picked.append(bases[i+1,...,ind])
            deriv = deriv_next[...,ind]


        bases_picked = np.stack(bases_picked,axis=0)
        bases_picked_unsliced = self.inv_reduce_all_cont(bases_picked)
        self.alphas = alphas
        return(bases_picked_unsliced)    

    def inv_reduce_all_disc(self,bases):
        """Simply calls the inversions for all discrete symmetry reductions in reversed order

        Args:
            bases (array): Symmetry reduced field
        
        Returns:
            array: one symmetry branch of the inverted field 
        """
        for op_num in reversed(range(self.total_op_disc)):
            bases = self.inv_reduce_disc(bases,op_num)         
        return(bases)
    
    def inv_reduce_disc(self,bases,op_num):
        """This inverts the discrete invariant polynomial for the discrete symmetry operation op_num

        Args:
            bases (array): symmetry reduced field
            op_num (int): operator index

        Returns:
            array: one branch of the possible symmetry copies from inverting the polynomials
        """
        tot_weight = self.disc_weights[op_num].copy()
        reducing_num_pow = self.get_reducing_num(np.take(bases,0,axis=-(1+op_num)),op_num,tot_weight)
        reducing_num = reducing_num_pow**(1./self.n[op_num])
        recon_field = []
        for ind in range(bases.shape[-(1+op_num)]):
            base = np.take(bases,ind,axis=-(1+op_num))
            if ind == self.n[op_num]-1:
                recon_field.append(base)
            else:
                recon_field.append(base/reducing_num**(self.n[op_num]-ind-1))
        out = sum(recon_field)
        return out

    def inv_reduce_all_cont(self,bases):
        """calls the inversion of the continuous symmetry reduction for all continuous symmetries in reversed order

        Args:
            bases (array): symmetry reduced field
        
        Returns:
            array: field with all continuous symmetries inverted
        """
        self.phase_speed_recon = []
        self.phase_recon = []
        for op_num in reversed(range(self.total_op_cont)):
            bases = self.inv_reduce_cont(bases,op_num)    
        if self.total_op_cont != 0:
            self.phase_speed_recon = self.phase_speed_recon[::-1]
            self.phase_recon = self.phase_recon[::-1]
        return(bases)
    
    def inv_reduce_cont(self,bases,op_num):
        """Inverts continuous symmetry reduction of continuous symmetry operation op_num by integrating the 
        reconstruction Eq.: C. W. Rowley and J. E. Marsden, “Reconstruction equations and the karhunen–lo`eve expansion for systems with symmetry,” Physica D 142, 1–19 (2000).

        Args:
            bases (array): symmetry reduced field
            op_num (int): operator index

        Returns:
            array: symmetry inverted field
        """
        phase_speed = self.reconstructio_eq(bases,op_num)
        self.phase_speed_recon.append(phase_speed)
        phase = integrate.cumulative_simpson(phase_speed, dx=self.dt,initial=self.phase_list[op_num][0])
        phase  = (phase + np.pi) % (2 * np.pi) - np.pi
        self.phase_recon.append(phase)
        out = self.cont_op[op_num](bases,phase)#[...,0,0])            
        return(out)
        
    def reconstructio_eq(self,field,op_num):
        """An implementation of the reconstruction Eq.: C. W. Rowley and J. E. Marsden, “Reconstruction equations and the karhunen–lo`eve expansion for systems with symmetry,” Physica D 142, 1–19 (2000).
        that finds the speed of the symmetry reducing phase from the symmetry reduced field. This requires knowledge of the underlying equations. 
        This could be learned using Neural Networks as well see e.g.: A. J. Linot and M. D. Graham, “Deep learning to discover and predict dynamics on an inertial manifold,” Phys. Rev. E 101, 062209 (2020).

        Args:
            field (array): symmetry reduced field
            op_num (int): operator index

        Returns:
            array: phase speed
        """
        try:
            rhs = self.RHS
            spatial_deriv = self.spatial_derivs[op_num]
        except:
            print('no RHS or spatial deriv provided for op: {}'.formmat(op_num))

        x = np.linspace(0,2*np.pi,field.shape[self.axes[op_num]],endpoint=False)
        dxC = np.cos(x)
        if self.orig_dims == 2:
            weight = 1.0
        else:
            weight = np.expand_dims(self.cont_weights[op_num].copy(),axis=self.axes[op_num])

        for i in range(field.ndim):
            if(i != self.axes[op_num]):
                dxC = np.expand_dims(dxC,axis=i)

        dxC = dxC * weight
        dtfield = rhs(field)
        dxfield = spatial_deriv(field)

        sum_axes = tuple(range(1,self.orig_dims))
        dtphi = np.sum(dxC*dtfield,axis=sum_axes)/np.sum(dxC*dxfield,axis=sum_axes)
        return dtphi
