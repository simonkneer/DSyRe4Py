from setuptools import setup, find_packages

setup(
    name="dsyre4py",
    version="1.0",
    packages=find_packages(),
    install_requires=['numpy==1.24.3',
                      'scipy==1.13.1',
                      'scikit-learn',
                      'torch',
                      'pytorch_lightning',
                      'matplotlib',
                      'h5py'],
)
