import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biapol-taurus",
    version="0.1.1",
    author="Robert Haase",
    author_email="robert.haase@tu-dresden.de",
    description="Bio-image analysis tools for the taurus cluster at TU Dresden",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiAPoL/biapol-taurus",
    packages=setuptools.find_packages(exclude=["docs"]),
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
    ],
)
