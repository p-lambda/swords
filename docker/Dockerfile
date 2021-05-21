FROM nvidia/cuda:10.1-base-ubuntu18.04

SHELL ["/bin/bash", "-c"]
ENV LANG C.UTF-8

# Install apt requirements
RUN apt-get -y update; apt-get -y install \
  build-essential \
  curl \
  wget

# Install Python
RUN apt-get -y install \
  python3 \
  python3-pip
RUN ln -s $(which python3) /usr/local/bin/python
RUN ln -s $(which pip3) /usr/local/bin/pip
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install Pytorch
RUN pip install --no-cache-dir torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir transformers==3.4.0
ENV TRANSFORMERS_CACHE /transformers_data
RUN python -c "from transformers import BertForMaskedLM; BertForMaskedLM.from_pretrained('bert-large-uncased')"
RUN python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-large-uncased')"
RUN python -c "from transformers import GPT2TokenizerFast; GPT2TokenizerFast.from_pretrained('gpt2')"
RUN chmod -R 755 $TRANSFORMERS_CACHE

# Install SpaCy
RUN pip install --no-cache-dir spacy==2.3.2
RUN python -m spacy download en_vectors_web_lg

# Install NLTK
RUN pip install --no-cache-dir nltk==3.5
ENV NLTK_CACHE /usr/share/nltk_data
RUN python -m nltk.downloader -d $NLTK_CACHE wordnet
RUN python -m nltk.downloader -d $NLTK_CACHE punkt
RUN python -m nltk.downloader -d $NLTK_CACHE averaged_perceptron_tagger
RUN python -m nltk.downloader -d $NLTK_CACHE wordnet_ic
RUN python -m nltk.downloader -d $NLTK_CACHE sentiwordnet
RUN chmod -R 755 $NLTK_CACHE

# Install Pattern
RUN apt-get -y install default-libmysqlclient-dev
RUN pip install --no-cache-dir pattern==3.6.0

# Install lightweight deps
RUN pip install --no-cache-dir intervaltree==3.1.0

# Install Jupyter
RUN pip install --no-cache-dir jupyter==1.0.0
RUN pip install --no-cache-dir matplotlib==3.3.2

# Install a fraction of the SWORDS library to download assets
ARG SWORDS_LIB_DIR=/swords
ENV SWORDS_ASSETS_DIR /swords/assets
RUN mkdir -p $SWORDS_LIB_DIR
RUN mkdir -p $SWORDS_ASSETS_DIR
WORKDIR $SWORDS_LIB_DIR
COPY setup.py setup.py
RUN mkdir -p $SWORDS_ASSETS_DIR/public/raw
COPY assets/public/raw/methods $SWORDS_ASSETS_DIR/public/raw/methods
RUN mkdir swords/
RUN pip install --no-cache-dir -e .

# Create user
RUN useradd -ms /bin/bash swords
RUN mkdir notebooks/
RUN chown -R swords $SWORDS_LIB_DIR
USER swords
