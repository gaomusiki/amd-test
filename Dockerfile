# from base image ngc 23.10
FROM nvcr.io/nvidia/pytorch:23.10-py3

# set workspace
WORKDIR /workspace

# install requirements
# COPY requirements.txt /workspace/
# RUN pip install --no-cache-dir -r requirements.txt

# install cython
# RUN pip install cython

# copy ref to workspace temporarily
COPY ref /workspace/ref

# compile all the .py files in ref/ to .so files
# and delete the source .py files
RUN cythonize -i -3 ref/**/*.py ref/*.py && \
    find ref/ -name "*.py" -type f -exec rm -f {} + && \
    find ref/ -name "*.c" -type f -exec rm -f {} + && \
    find ref/ -type d -name "__pycache__" -exec rm -rf {} +

# add ref to python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# delete the cython and other unnecessary packages
# RUN pip uninstall -y cython

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*