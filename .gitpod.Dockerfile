# This will pull the official Gitpod `vnc` image
# which has much of what you need to start
FROM gitpod/workspace-full-vnc

USER gitpod

# Install wxPython dependencies
RUN pyenv install 3.6.4 && \
    pyenv global 3.6.4

# Install wxPython
RUN pip install torch torchvision flask

