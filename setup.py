import setuptools

setuptools.setup(
    name='dlhlp-lib',               
    version='0.2.0',
    packages=['dlhlp_lib'],   
    include_package_data=True,  
    exclude_package_date={'':['.gitignore']},
    python_requires='>=3.8',
    install_requires=[
        'tqdm',
        'librosa',
        'numpy',
        'tqdm',
        'resemblyzer',
        'tgt',
        'pyworld'
    ]
)
