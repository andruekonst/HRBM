import setuptools


long_description = ""


setuptools.setup(
    name="hrbm",
    version="0.0.1",
    author="Andrei V. Konstantinov",
    author_email="andrue.konst@gmail.com",
    description="Hyper-Rectangles Base Models Boosting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andruekonst/hrbm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    # install_requires=install_requirements
)
