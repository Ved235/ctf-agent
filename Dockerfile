FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive

# Fix Kali GPG Keys
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --no-install-recommends --allow-unauthenticated kali-archive-keyring && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Core system + Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        curl wget git ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install kali-linux-headless (brings most pentesting tools + dependencies)
# This includes: nmap, nikto, dirb, gobuster, sqlmap, netcat, ssh, etc.
RUN apt-get update && \
    echo "console-setup console-setup/variant select Latin1 and Latin5 - western Europe and Turkic languages" | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        kali-linux-headless \
        seclists \
        burpsuite \
        default-jre \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Metasploit
 RUN curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > /tmp/msfinstall && \
     chmod 755 /tmp/msfinstall && \
     /tmp/msfinstall && \
     rm /tmp/msfinstall

# Allow pip into system site-packages in this dev image.
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "break-system-packages = true" >> /root/.pip/pip.conf

# Workspace where your host repo is bind-mounted.
WORKDIR /workspace

# Install CAI 
RUN pip3 install  --ignore-installed cai-framework

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import cai; print('OK')" || exit 1

# Keep container alive for interactive dev (`docker compose exec ...`).
CMD ["bash", "-lc", "tail -f /dev/null"]
