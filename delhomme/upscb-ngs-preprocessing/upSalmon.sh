#! /bin/bash
mkdir -p /build && cd /build && \
wget https://github.com/COMBINE-lab/salmon/releases/download/v1.1.0/salmon-1.1.0_linux_x86_64.tar.gz && \
tar -zxf salmon-1.1.0_linux_x86_64.tar.gz && cp salmon-latest_linux_x86_64/bin/salmon /usr/local/bin/ && \
cp salmon-latest_linux_x86_64/lib/lib* /usr/local/lib/
