FROM ubuntu
MAINTAINER nicolas.delhomme@umu.se
RUN apt update && apt install -y build-essential software-properties-common wget
RUN wget -q https://xpra.org/gpg.asc -O- | APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key add -
RUN add-apt-repository "deb https://xpra.org/ bionic main"
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y xpra
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y arb
RUN useradd -m -s /bin/bash -g 116 arb -u 13971
VOLUME /mnt/ext/data /mnt/ext/pt-server
EXPOSE 3333
RUN chgrp -R arb /var/lib/arb
USER arb
CMD xpra start --bind-tcp=0.0.0.0:3333 --start=arb --html=on --mdns=no --pulseaudio=no --tcp-auth=password:value=CHANGEME --daemon=no
