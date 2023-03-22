"""
ITP Group MA3
This file is to test requirements of the project,
the project can not be used on the computer if the requirement is not needed

"""


def version_test(print_version=False):
    import tensorflow, keras, numpy, sklearn, scipy, matplotlib, seaborn, pandas, xgboost, lightgbm
    libraries = {tensorflow: '2.2.0', keras: '2.2.0', numpy: '1.22.0', sklearn: '1.1.0', scipy: '1.10.0',
                 matplotlib: '3.4.0', seaborn: '0.10.2', pandas: '1.1.0', xgboost: '1.4.0', lightgbm: '3.2.0'}
    for library in libraries:
        if print_version is True:
            print(library.__version__)
        assert library.__version__ >= libraries[library], \
            f'{library} version test failed, requirement: {libraries[library]}, current: {library.__version__}'

    print('version test over')
    return

# version_test()


