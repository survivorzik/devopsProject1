import sys 
import logging

def error_message_detail(error,error_detail:sys):
    """Prints the error message and the line number where the error occured"""
    _,_,exc_tb=error_detail.exc_info() # gives information about error where it has occured and in which line it has occured 
    filename=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line numer [{1}] error [{2}]".format(
        filename,exc_tb.tb_lineno,str(error)
        
    )
    return error_message
    
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        # error_message_detail(error_message,error_detail)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message


