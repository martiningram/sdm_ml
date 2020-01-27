FROM continuumio/miniconda3

# Copy the local repos -- could clone these eventually
COPY sdm_ml /opt/sdm_ml
COPY ml_tools /opt/ml_tools
COPY svgp /opt/svgp

# This is private data -- see README on how to obtain it
COPY csv_bird_data /opt/csv_bird_data

# Install the repositories
RUN cd /opt/ml_tools && pip install -e .
RUN cd /opt/sdm_ml && pip install -e .
RUN cd /opt/svgp && pip install -e .

# Install the conda requirements
RUN cd /opt/sdm_ml && conda install --yes --file requirements.txt

# Install the R requirements
RUN R -e "install.packages(c('dismo', 'gbm'), repos = 'http://cran.us.r-project.org')"
