# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The Dockerfile is provided as reference. Please review the redistribution
# terms of the Mojo license in Section 1 (https://www.modular.com/legal/mojo)
# prior to distributing pre-built container images.
# ===----------------------------------------------------------------------=== #

# Example command line:
# Use no-cache to force docker to rebuild layers of the image by downloading the SDK from the repos
# ./build-image.sh --mojo-version
#

FROM ubuntu:20.04

ARG DEFAULT_TZ=America/Los_Angeles
ENV DEFAULT_TZ=$DEFAULT_TZ

RUN apt-get update \
   && DEBIAN_FRONTEND=noninteractive TZ=$DEFAULT_TZ apt-get install -y \
   tzdata \
   vim \
   nano \
   sudo \
   curl \
   wget \
   git && \
   rm -rf /var/lib/apt/lists/*

# Download the latest version of minicoda py3.8 for linux x86/x64.
RUN curl -fsSL https://repo.anaconda.com/miniconda/$( wget -O - https://repo.anaconda.com/miniconda/ 2>/dev/null | grep -o 'Miniconda3-py38_[^"]*-Linux-x86_64.sh' | head -n 1) > /tmp/miniconda.sh \
       && chmod +x /tmp/miniconda.sh \
       && /tmp/miniconda.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:$PATH
RUN conda init

RUN pip install \
        jupyterlab \
        ipykernel \
        matplotlib \
        ipywidgets

RUN curl -s https://get.modular.com | sh -
RUN modular install mojo

ARG MODULAR_HOME="/root/.modular"
ENV MODULAR_HOME=$MODULAR_HOME
ENV PATH="$PATH:$MODULAR_HOME/pkg/packages.modular.com_mojo/bin"

# Change permissions to allow for Apptainer/Singularity containers
RUN chmod -R a+rwX /root

RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
CMD ["jupyter", "lab", "--ip='*'", "--NotebookApp.token=''", "--NotebookApp.password=''","--allow-root"]
