import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Load requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="statspal",
    version="0.1.0",
    author="J. Nathan White",
    description="A lightweight statistical distribution inference engine using ONNX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwhitebored/statspal",
    packages=setuptools.find_packages(), # Automatically finds the 'statspal' directory
    install_requires=required, # Uses the requirements.txt list
    
    # CRUCIAL: Tells setuptools to look for non-code files (like the ONNX model)
    include_package_data=True,
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.8',
)
