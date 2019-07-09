import os
import sys
import socket
import pathlib
import logging
log = logging.getLogger(__name__)

class CBridge:
    def __init__(self, host: str, port: str, verbose: bool):
        """
        A bridge to communicate with the C Code.

        Parameters
        ----------
        host: str
            Host address
        port: srt
            Connection port
        verbose: bool
            Turn on verbose mode
        """
        self.host = host
        self.port = port
        self.verbose = verbose
        

    def setup_server(self) -> None:
        """
        Initialize a communication with sockets
        """
        # Initialize the socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind host to port
        sock.bind((self.host, self.port))
        # Start listening with a maximum of 1 queued connection
        sock.listen(backlog=1)
        # Begin communication
        conn, addr = sock.accept()
        
        if self.verbose:
            log.info("Established connection to NEUZZ execution module at {}".format(str(addr)))
        
        



