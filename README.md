# **DSyRe4Py** - ( •_•) 🥧 - **D**ifferentiable **Sy**mmetry **Re**duction **for** **Py**thon
DSyRe4Py - ( •_•) 🥧 (pronounced *desire for pie*) is a tool to perform both continuous and discrete symmetry reduction for time-continuous dynamical systems. 
Continuous symmetry reduction is based on the first Fourier slice of Budanur et al. (2015) 
and the extension by Marensi et al. (2022), while 
discrete symmetry reduction is based on the generalization made to invariant polynomials in Kneer (2025). While the First Fourier mode slice can be considered established at this point, invariant polynomials still remain an enigma for most of the community, despite the fact that - in contrast to fundamental domain methods - they do **not** introduce discontinuities into the trajectories when reducing the symmetries. 

The general idea of **handcrafted** invariant polynomials was pioneered by Budanur and Cvitanović (2016) for $2$-cyclic symmetries of spatially extended systems. Later, Kneer and Budanur (2025) expanded this to the $4$-cyclic, discrete symmetry in Kolmogorov flow by once again creating a set of **problem specific** polynomials and shoring up the belief that such polynomials should be craftable for any discrete, cyclic symmetry.

However, as has been made clear by comments from various colleagues on top of the almost decade long break in developments, discrete symmetry reduction needed to be made more accessible to researchers by providing invariant polynomials **without** the need to handcraft them anew for each problem. 

This is precisely the goal of this piece of software: All one needs to perform symmetry reduction now is to provide a python function that represents the action of the symmetry operator on your dataset and the rest will be handled in the background. 

## Installation

DSyRe4Py requires Python 3.8,3.9,3.10 or 3.11. 
To install, simply clone the repository with 
```
git clone https://github.com/simonkneer/DSyRe4Py.git
```
and install using pip by running
```
pip install .
```
in the DSyRe4Py directory. pip will then install the required python packages and detect eventual version conflicts. Please see setup.py for these. Change these at your own risk!
## Usage
DSyRe4Py is built around the ```symmred``` class which contains all necessary operations to perform symmetry reduction for both continuous and discrete symmetries.
### Operator Definition
To be able to reduce the symmetries DSyRe4Py needs the symmetry operations as functions.
As an example, let us inspect the symmetries of Kolmogorov Flow with corresponding exemplary functions:

$\mathcal{T}(s)\omega(x,y)\rightarrow\omega(x-s,y)$
```
def translate(u,s=0):
    u_fft = np.fft.rfft(u.copy(),axis=(1))
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
```

$C_4$: $\mathcal{S}\omega(x,y)\rightarrow-\omega(-x,y+\pi/2)$
```
def shift_reflect(u):
    u_fft = np.fft.rfft(u.copy(),axis=(2))
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
```

$C_2$: $\mathcal{R}\omega(x,y)\rightarrow\omega(-x,-y)$
```
def rotate(u):
    u_fft = np.fft.rfft2(u.copy(),axes=(1,2))
    u_out_fft = np.conjugate(u_fft)
    return(np.fft.irfft2(u_out_fft,axes=(1,2))
)
```
$\color{red}{\text{Important:}}$
Note how we expand the dimensions of $k_x$ and $k_y$ above. This is because additional dimensions will be appended during the symmetry reduction process, e.g. a field of size ```(1000,64,64)``` will become ```(1000,64,64,4,2)``` in the Kolmogorov case. Thus make sure that your symmetry operators work on the correct axes to guarantee proper functionality.
### Initialize ```symmred``` class

Having defined the symmetry operators and knowing the cyclicity (i.e. how often you need to apply the discrete operators to get the original field back) we can initialize a ```symmred``` class with 
```
from dsyre4py import symmred
sr = symmred([translate],[1],[shift_reflect,rotate],[4,2],data_shape,weight_path=weight_path)
```
where the first two lists contain the names of the **continuous** symmetry operators and the axes in the datasets in which the continuous symmetries act. In our example the zeroth axis of the data array would be time, the first axis the $x$ direction and the second axis the $y$ direction. The third and fourth list are the discrete symmetry operators and their respective cyclicities.
If your system only has continuous or discrete symmetries you simply leave the other lists empty.
Finally, we need the original input shape of the data we want to reduce (most probably just with ```data.shape()```) and the location+name we want to save the symmetry reducing weights in.

### Reduce Symmetries
To reduce the symmetries we simply call
```
reduced = sr.reduce_all(data)
```
which will optimize the weights for symmetry reduction (if they don't exists already) and return an array of the shape of ```data``` + ```(cyclicity_1, cyclicity_2,...)``` with all symmetries reduced.
If there are only continuous symmetries present, or you only want to reduce the continuous symmetries you call 
```
cont_reduced = sr.cont_reduce(data)
```
instead. You can now proceed to do whatever it is you wanted to do with the dataset, e.g. modelling, dimensionality reduction searching for recursions, classification ...
### Invert Symmetry Reduction
Say you modelled your dataset in the symmetry reduced state space, and now you want to take your model back to the input state space.
If we don't care about symmetry orientations we can simply
call
```
reduced_inverted  = sr.inv_reduce_all_disc(reduced)
```
which will return an arbitrary symmetry copy in the First Fourier mode slice. Additionally, we provide a function that picks the symmetry copy that simply minimizes the difference from the unreduced data (```data```) with 
```
reduced_inverted_picked  = sr.inv_reduce_all_static(reduced,data)
```
For dynamical modelling we also provide the method used in Kneer and Budanur (2025) which matches the temporal derivatives to pick the correct discrete symmetry copy (see below). If there is a continuous symmetry present we will also need to provide a way to 
invert the continuous symmetry reduction. For this the reconstruction equation by Rowley and Marsden (2000) is used. Since this method requires the temporal derivative as well as the one in the translational direction we need to change the initialization of ```symmred``` to 
```
sr = symmred([translate],[1],[shift_reflect,rotate],[4,2],data_shape,weight_path=weight_path,RHS=get_deriv_time,spatial_derivs=[get_deriv_x],dt=dt)
```
**before** reducing the symmetries. Here ```RHS=get_deriv_time``` is a function returning a set of temporal derivative for a number of input states, i.e. what your numerical solver integrates wrapped so that it does it for all states in an array, not just one. ```spatial_derivs``` is a list containing the spatial derivatives for the translated directions. If you have more than one continuous symmetry make sure that the ordering here is the same as in the original list defining the symmetry operators. Finally, ```dt``` is the sampling time step.
With this we can then invert the symmetries using 
```
reduced_inverted = sr.inv_reduce_all_dynamic(reduced,data)
```
where you still need to pass ```data``` but only as an **initial condition** for the correct symmetry copy and the phase of the continuous symmetry.
### Examples
We provide three exemplary fluid dynamics datasets exhibiting symmetries.
These are, in order of increasing complexity: The flow past a cylinder at $\text{Re}=100$, with a $2$-cyclic reflection symmetry; Kolmogorov Flow at $\text{Re}=14.3$, 
with a continuous symmetry, a $4$ and a $2$-cyclic discrete symmetry; Kolmogorov Flow at $\text{Re}=14.4$, with the same symmetries but chaotic dynamics. See Kneer and Budanur (2025) for simulation details. For each of these examples we provide a script to perform symmetry reduction followed by PCA on the original input data and the reduced data - ```reduce_and_pca.py```. We are aware of the shortcomings of PCA but since it is an analytically solvable problem it serves to illustrate how the method can aid in dimensionality reduction. When running the scripts for the first time the data is downloaded automatically from 
[Zenodo](https://zenodo.org/records/15195938?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZlZDQ4OThkLWIyMDgtNGMyNS1iYzg4LTM5NjRlYjlmNjc4OSIsImRhdGEiOnt9LCJyYW5kb20iOiIyZjNmZjE1YWQ2MDY2NGVmOTRmYTAxZjUzODAyYzgyNiJ9.kcTpMgIamOcMblmFemkMjf7yl3sN8Cip8rmd3wB26J6g6lgy6CZCwFB1WCznbaNA10XfwIo4ujJEaNGI8DworQ) (File sizes: Cylinder - 3.2 GB; Kolmogorov 14.3 - 16.4 MB; Kolmogorov 14.4 - 1.3 GB). 

$\color{red}{\text{I suggest you try running the Kolmogorov 14.3 example first, due to its small size.}}$
## How it works
DSyRe4Py first reduces all continuous symmetries followed by the discrete symmetries. 
Where appropriate the corresponding functions of the ```symmred``` class are noted along with their mathematical counterparts.
For notational ease define the inner product as  
```math
\langle F(x,y),G(x,y) \rangle_x = \int_xF(\xi,y)G(\xi,y)\text{d}\xi
```
or in the discretized case
```math
\langle F(\mathbf{x},\mathbf{y}),G(\mathbf{x},\mathbf{y})\rangle_x = \sum_iF(x_i,\mathbf{y})G(x_i,\mathbf{y}).
```
Let us first recap how the First Fourier mode slice works.
### First Fourier Mode Slice
#### Symmetry Reduction - ```cont_reduce```
Assuming we have some $2$-D field $u(t,x,y)$ exhibiting a continuous symmetry in $x\in[0,2\pi)$ with corresponding operator $\mathcal{T}(s)u(t,x,y)=u(t,x-s,y)$ then 
```math
\bar{u}(t,x,y) = \bar{u}\Big(u(t,x,y)\Big) = \mathcal{T}(\phi_\mathcal{T}(t))u(t,x,y)
```
with 
```math
\phi_\mathcal{T}(t) = \phi_\mathcal{T}\Big(u(t,x,y)\Big) = \arg\left(\left\langle f\left(y\right),U(t,k_x=1,y)\right\rangle_y\right)
```
is a symmetry reduction of the continuous symmetry. Here, $U(t,k_x=1,y)$ is the Fourier transform in the $x$ direction evaluated at $k_x=1$ and $f(y)$ is a weighting function to reduce the temporal gradient of $\phi_\mathcal{T}(t)$. In Marensi et al. (2022) $f(y)$ was chosen as a combination of Chebyshev Polynomials, which would be easy to implement in this package as well.
For simplicity, we instead use
```math
f(y)=\sum_{i=0}^N c_i\delta(y'_i),
```
where $c_i$ are some **real**-valued coefficients, $\delta$ is the Dirac delta function and
```math
y'=\underset{y}{\overset{N}{\text{argmax}}}\left(\sum_t\left|U(t,k_x=1,y)\right|\right)
```
are a number, $N$, of $y$-coordinates with the largest temporal average of the absolute value of $U(t,k_x=1,y)$. In practice our data is of course discretely sampled, meaning we are simply looking for the grid points in $y$ that have the largest amplitude on average.
We then optimize the weights, $c_i$, using PyTorch with the loss function
```math
L = \max_t\left(\sin\left(\frac{\phi_\mathcal{T}(t)-\phi_\mathcal{T}(t+t_i)}{2}\right)^2\right)+\left(1-\sqrt{\sum_i c_i^2}\right)^2,
```
where $t_i$ is the integration time step. The first term in this loss minimizes the maximal jump in symmetry reducing phase from one time step to the next while ignoring jumps of $2\pi$. In practice, we will not be able to process $\max_t$ for the whole trajectory length but only batches of data. The second term deals with the marginal direction in the optimization introduced by $\arg$, since it returns the same value no matter the magnitude of its input.
#### Inverse Symmetry Reduction - ```inv_reduce_cont```
It was shown by Rowley and Marsden (2000) that the symmetry reducing phase can be recovered by integrating the reconstruction equation (```reconstructio_eq```) which in our example reads:
```math 
\frac{\partial \phi_\mathcal{T}(t)}{\partial t} = \frac{\left\langle f(y)\cos(x),\frac{\partial u}{\partial t}|_{u=\bar{u}}\right\rangle_{x,y} }{\left\langle f(y)\cos(x),\frac{\partial \bar{u}}{\partial x}\right \rangle_{x,y} }.
```
### Invariant Polynomials
#### Symmetry Reduction - ```disc_reduce```

Assuming the $2$-D field $\omega(t,x,y)$ has an $n$-cyclic symmetry $G$ with 
group elements $\{\mathcal{I},\mathcal{G}^1,\dots,\mathcal{G}^{n-1}\}$, where $\mathcal{G}$ is the associated symmetry operator and $\mathcal{I}$ the identity operator. We will start reducing this symmetry by projecting $u(t,x,y)$ onto the basis for the irreducible representation of $G$ (```bases```) with 
```math
 u_j(t,x,y) = u_j\Big(u(t,x,y)\Big)= \frac{1}{n}\sum_{i=0}^{n-1}\alpha_n^{-ji}F\left(\mathcal{G}^iu(t,x,y)\right) \quad j\in\{1,2,\dots,n\},
```
where $\alpha_n$ is the $n$-th principal root of unity and $F$ is some previous transformation such as the continuous symmetry reduction ($F=\bar{u}$) or another discrete symmetry reduction. Note that
when $n=2$ this is simply a decomposition of the field into symmetric and antisymmetric parts and in general this is simply a generalization of the discrete Fourier transform. Also note that 
```math 
u(t,x,y) = \sum_j u_j(t,x,y).
```
These decomposed fields then transform under the symmetry operator like
```math 
u_j\Big(\mathcal{G}u(t,x,y)\Big) = \alpha_n^j u_j\Big(u(t,x,y)\Big).
```
To generate a complex, symmetry reducing unit number (```get_reducing_num```), similar to the phase in the continuous case, we form 
```math
p_\mathcal{G}(t) = p_\mathcal{G}\Big(u(t,x,y)\Big) = \frac{\left \langle g\left(x,y\right),u_{\color{red}{1}}(t,x,y)\right \rangle_{x,y}}{\left|\left \langle g\left(x,y\right),u_{\color{red}{1}}(t,x,y)\right \rangle_{x,y}\right|},
```
where $g(x,y)$ is a weighting function to keep the temporal gradients small again. Since $p_\mathcal{G}$ is dependent on symmetry operations through $u_1$ it transforms like
```math
p_\mathcal{G}\Big(\mathcal{G}u(t,x,y)\Big) = \alpha_n p_\mathcal{G}\Big(u(t,x,y)\Big)
```
and we can perform symmetry reduction on (almost) **all** $u_j$ with 
```math
\bar{u}_j\Big(u(t,x,y)\Big) = \Big(p_\mathcal{G}(t)\Big)^{n-j}u_j(t,x,y)\quad j\in\{1,2,\dots,n-1\}, 
```
since
```math
\bar{u}_j\Big(\mathcal{G}u(t,x,y)\Big) = \Big(\alpha_np_\mathcal{G}(t)\Big)^{n-j}\alpha_n^{j}u_j(t,x,y) = \Big(p_\mathcal{G}(t)\Big)^{n-j}u_j(t,x,y) = \bar{u}_j\Big(u(t,x,y)\Big). 
```
Note that $u_{n}$ is already invariant w.r.t to the symmetry so we don't need to multiply it with the reducing number.
We can recover $p_\mathcal{G}^n$ (```get_reducing_num```) from $\bar{u}_1$ with 
```math 
\frac{\left\langle g\left(x,y\right),\bar{u}_{{1}}(t,x,y)\right\rangle_{x,y} }
{\left|\left\langle g\left(x,y\right),\bar{u}_{{1}}(t,x,y)\right\rangle_{x,y}\right|}=\frac{\left\langle g\left(x,y\right),{u}_{{1}}(t,x,y)p_\mathcal{G}^{n-1}(t)\right\rangle_{x,y} }{\left|\left\langle g\left(x,y\right),{u}_{{1}}(t,x,y)p_\mathcal{G}^{n-1}(t)\right\rangle_{x,y} \right|}=p_\mathcal{G}^n(t), 
```
which gives $n$ possible solutions for $p_\mathcal{G}^n$ corresponding to the $n$ symmetry copies. With these $n$ choices of $p_\mathcal{G}$ we can then recover the $n$ possible symmetry copies of the states in the input space. Before we elaborate on how one can choose the correct symmetry copy, let us quickly go over the weighting function we use to find $p_\mathcal{G}$. As for the continuous case we chose $g(x,y)$ to be
```math
g(x,y)=\sum_{i=0}^N c_i\delta(x'_i)\delta(y'_i)
```
where $c_i$ are some **complex**-valued coefficients, $\delta$ is the Dirac delta function and
```math
(x,y)'=\underset{x,y}{\overset{N}{\text{argmax}}}\left(\sum_t\left|u_1(t,x,y)\right|\right)
```
are a number, $N$, of $(x,y)$-coordinates with the largest temporal average of the absolute value of $u_1(t,x,y)$. Similar to above we optimize $c_i$
with 
```math
L = \max_t\left(\sin\left(\frac{\arg\left(p_\mathcal{G}(t)\right)-\arg\left(p_\mathcal{G}(t+t_i)\right)}{2}\right)^2\right)+\left(1-\sqrt{\sum_i \text{Re}(c_i)^2}-\sqrt{\sum_i \text{Im}(c_i)^2}\right)^2.
```
#### Trajectory Reconstruction - ```inv_reduce_all_dynamic```
To choose the correct symmetry copy we can follow the approach from Kneer and Budanur (2025) where we match the temporal derivatives in time. 
Let $u(t=0,x,y)$ be an initial "measured" state. At every point, $m$, along our symmetry reduced trajectory the inversion of the polynomials generates $n$ possible symmetry copies
$\mathcal{G}^j u(t=m \Delta t,x,y)$ with $j \in {0, 1, 2, \ldots, n}$ from which we have to choose the correct one with $j=j^*$. 
At $m=0, 1$, we choose
```math 
j^* = \underset{j}{{\text{argmin}}}\left(\| u(m\Delta t,x,y) - \mathcal{G}^{j}u(m\Delta t,x,y) \|\right).
``` 
For the subsequent time steps, we estimate 
```math
    f_{m,j}=\mathcal{G}^jf(m\Delta t,x,y) \approx \frac{\mathcal{G}^ju(m\Delta t,x,y)-\mathcal{G}^{j^*}u((m-1)\Delta t,x,y)}{\Delta t}
```
and choose 
```math
    j^* = \underset{j}{{\text{argmax}}} \left( \frac{\left\langle{f_{m,j},f_{j^*,m-1}}\right\rangle_{x,y}}
    {\sqrt{\left\langle{f_{m,j},f_{m,j}}\right\rangle_{x,y}\left\langle{f_{m-1,j^*},f_{m-1,j^*}}\right\rangle_{x,y}}}\right).
```
## Inheriting DSyRe4Py - ( •_•) 🥧
I am actively looking for someone to inherit DSyRe4Py - ( •_•) 🥧 since I have left academia. If you know someone who might be interested or are yourself interested please let me know - or just fork it. 

## Citation
If you use DSyRe4Py - ( •_•) 🥧 please consider a citation of the 
code itself
```
@misc{kneer2025dsyre4py,
  author       = {Simon Kneer},
  title        = {DSyRe4Py - ( •_•) 🥧 - Differentiable Symmetry Reduction for Python},
  month        = apr,
  year         = 2025,
  url          = {https://github.com/simonkneer/DSyRe4Py}
}
```
alongside the methods 
```
@article{budanur2015reduction,
 author = {N. B. Budanur and P. Cvitanovi{\'{c}} and R. L. Davidchack and E. Siminos},
 doi = {10.1103/physrevlett.114.084102},
 journal = {Phys. Rev. Lett.},
 number = {8},
 pages = {084102},
 publisher = {American Physical Society ({APS})},
 title = {Reduction of {SO}(2) Symmetry for Spatially Extended Dynamical Systems},
 url = {https://doi.org/10.1103%2Fphysrevlett.114.084102},
 volume = {114},
 year = {2015}
}

@article{marensi2022symmetryreduced,
 author = {E. Marensi and G. Yaln{\i}z and B. Hof and N. Budanur},
 doi = {10.1017/jfm.2022.1001},
 journal = {J. Fluid Mech.},
 month = {dec},
 pages = {A10},
 publisher = {Cambridge University Press ({CUP})},
 title = {Symmetry-reduced dynamic mode decomposition of near-wall turbulence},
 url = {https://doi.org/10.1017%2Fjfm.2022.1001},
 volume = {954},
 year = {2022}
}

@phdthesis{kneer2025symmetryreduction,
  author  = {S. Kneer},
  title   = {Symmetry-Reduction for Reduced Order Modelling of Fluid Flows},
  school  = {TU Dresden},
  year    = {2025},
  url = {https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-964947}
}
```
## References
N. B. Budanur, P. Cvitanović, R. L. Davidchack, and E. Siminos,
"Reduction of SO(2) symmetry for spatially extended dynamical
systems", Phys. Rev. Lett. 114, 084102 (2015).

E. Marensi, G. Yalnız, B. Hof, and N. Budanur, “Symmetry-
reduced dynamic mode decomposition of near-wall turbulence,”
J. Fluid Mech. 954, A10 (2022).

S. Kneer, "Symmetry-Reduction for Reduced Order Modelling of Fluid Flows",
Dissertation, TU Dresden (2025), URL: [https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-964947](https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-964947).

N. B. Budanur and P. Cvitanović, "Unstable manifolds of relative
periodic orbits in the symmetry-reduced state space of the kuramoto–sivashinsky system", J. Stat. Phys. 167, 636–655 (2016).

S. Kneer and N. B. Budanur, "Learning the dynamics of symmetry-reduced chaotic attractors from data", Chaos (under revision).

C. W. Rowley and J. E. Marsden, “Reconstruction equations
and the karhunen–loève expansion for systems with symmetry,”
Physica D 142, 1–19 (2000).


