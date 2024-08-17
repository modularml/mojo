# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Olivier Benz
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

ARG BUILD_ON_IMAGE=glcr.b-data.ch/mojo/base
ARG MOJO_VERSION=nightly
ARG LLVM_VERSION

FROM ${BUILD_ON_IMAGE}:${MOJO_VERSION} as mojo

ARG DEBIAN_FRONTEND=noninteractive

ARG BUILD_ON_IMAGE
ARG LLVM_VERSION

ENV PARENT_IMAGE=${BUILD_ON_IMAGE}:${MOJO_VERSION} \
    PARENT_IMAGE_BUILD_DATE=${BUILD_DATE} \
    LLVM_VERSION=${LLVM_VERSION}

ENV PATH=${LLVM_VERSION:+/usr/lib/llvm-}${LLVM_VERSION}${LLVM_VERSION:+/bin:}$PATH

RUN dpkgArch="$(dpkg --print-architecture)" \
  ## Ensure that common CA certificates
  ## and OpenSSL libraries are up to date
  && apt-get update \
  && apt-get -y install --no-install-recommends --only-upgrade \
    ca-certificates \
    openssl \
## Install Modular
  && apt-get -y install --no-install-recommends \
    apt-transport-https \
  && curl -s https://dl.modular.com/public/installer/setup.deb.sh | bash - \
  && apt-get -y install --no-install-recommends \
    modular \
  && if [ "$MOJO_VERSION" = "nightly" ]; then \
    ## Update Mojo nightly
    modular update nightly/mojo; \
  fi \
  ## Clean up
  && rm -rf /root/.cache \
    /root/.ipython \
    /root/.lldb \
    /root/.local \
## Install tools for buildig and testing the Mojo standard library
  && if [ -n "$LLVM_VERSION" ]; then \
    ## Install LLVM
    apt-get -y install --no-install-recommends \
      lsb-release \
      software-properties-common; \
    curl -sSLO https://apt.llvm.org/llvm.sh; \
    chmod +x llvm.sh; \
    ./llvm.sh "$LLVM_VERSION"; \
    ## Clean up
    rm -rf llvm.sh \
      /root/.ssh \
      /root/.wget-hsts; \
  else \
    ## Install FileCheck
    apt-get -y install --no-install-recommends llvm-14-tools; \
    cp -a /usr/lib/llvm-14/bin/FileCheck /usr/local/bin/FileCheck; \
    ## Clean up
    apt-get -y purge llvm-14-tools; \
    apt-get -y autoremove; \
  fi \
  ## Install lit
  && pip install --no-cache-dir lit \
  ## Install pre-commit
  && pip install --no-cache-dir pre-commit \
  ## Install entr
  && apt-get -y install --no-install-recommends entr \
## Dev Container only
  ## Install hadolint
  && case "$dpkgArch" in \
    amd64) tarArch="x86_64" ;; \
    arm64) tarArch="arm64" ;; \
    *) echo "error: Architecture $dpkgArch unsupported"; exit 1 ;; \
  esac \
  && apiResponse="$(curl -sSL \
    https://api.github.com/repos/hadolint/hadolint/releases/latest)" \
  && downloadUrl="$(echo "$apiResponse" | grep -e \
    "browser_download_url.*Linux-$tarArch\"" | cut -d : -f 2,3 | tr -d \")" \
  && echo "$downloadUrl" | xargs curl -sSLo /usr/local/bin/hadolint \
  && chmod 755 /usr/local/bin/hadolint \
  ## Create backup of root directory
  && cp -a /root /var/backups \
  ## Clean up
  && rm -rf /tmp/* \
  && rm -rf /var/lib/apt/lists/*

FROM ${BUILD_ON_IMAGE}:${MOJO_VERSION} as files

RUN mkdir /files

COPY conf/shell /files
COPY scripts /files

  ## Ensure file modes are correct
RUN find /files -type d -exec chmod 755 {} \; \
  && find /files -type f -exec chmod 644 {} \; \
  && find /files/usr/local/bin -type f -exec chmod 755 {} \; \
  ## Copy skeleton files for root
  && cp -r /files/etc/skel/. /files/root \
  ## except .bashrc and .profile
  && bash -c 'rm -rf /files/root/{.bashrc,.profile}' \
  && chmod 700 /files/root

FROM docker.io/koalaman/shellcheck:stable as sci

FROM mojo

## Copy files as late as possible to avoid cache busting
COPY --from=files /files /

## Copy shellcheck as late as possible to avoid cache busting
COPY --from=sci --chown=root:root /bin/shellcheck /usr/local/bin

ARG DEBIAN_FRONTEND=noninteractive

## Update Modular settings
ARG MODULAR_TELEMETRY_ENABLE
ARG MODULAR_TELEMETRY_LEVEL
ARG MODULAR_CRASHREPORTING_ENABLE

RUN if [ -n "$MODULAR_TELEMETRY_ENABLE" ]; then \
    ## Enable telemetry level 2
    modular config-set telemetry.enabled=true; \
    modular config-set telemetry.level="$MODULAR_TELEMETRY_LEVEL"; \
  fi \
  && if [ -n "$MODULAR_CRASHREPORTING_ENABLE" ]; then \
    ## Enable crash reporting
    modular config-set crash_reporting.enabled=true; \
  fi \
  ## Change ownership and permission for MODULAR_HOME
  && chown -R :1000 "$MODULAR_HOME" \
  && chmod -R g+w "$MODULAR_HOME" \
  && chmod -R g+rx "$MODULAR_HOME/crashdb"

## Update environment
ARG USE_ZSH_FOR_ROOT
ARG LANG
ARG TZ

ARG LANG_OVERRIDE=${LANG}
ARG TZ_OVERRIDE=${TZ}

ENV LANG=${LANG_OVERRIDE:-$LANG} \
    TZ=${TZ_OVERRIDE:-$TZ}

  ## Change root's shell to ZSH
RUN if [ -n "$USE_ZSH_FOR_ROOT" ]; then \
    chsh -s /bin/zsh; \
  fi \
  ## Info about timezone
  && echo "TZ is set to $TZ" \
  ## Add/Update locale if requested
  && if [ "$LANG" != "en_US.UTF-8" ]; then \
    sed -i "s/# $LANG/$LANG/g" /etc/locale.gen; \
    locale-gen; \
  fi \
  && update-locale --reset LANG="$LANG" \
  ## Info about locale
  && echo "LANG is set to $LANG"

## Unset environment variable BUILD_DATE
ENV BUILD_DATE=
