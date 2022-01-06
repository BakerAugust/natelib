#
# Analysis Example
# Get Device List
#
# This analysis retrieves the device list of your account and print to the console.
# There are examples on how to apply filter.
#
# Environment Variables
# In order to use this analysis, you must setup the Environment Variable table.
#
# account_token: Your account token
#
# Steps to generate an account_token:
# 1 - Enter the following link: https://admin.tago.io/account/
# 2 - Select your Profile.
# 3 - Enter Tokens tab.
# 4 - Generate a new Token with Expires Never.
# 5 - Press the Copy Button and place at the Environment Variables tab of this analysis.

from tago import Account
from tago import Analysis
from tago import Services
from tago import Device
from datetime import timedelta, datetime
from dateutil.parser import parse
from pytz import timezone

MAX_HOURS_SINCE_LAST_MEASUREMENT = 12
# NOTIFICATION_RECIPIENT_ID = '603dbebb8448b30010a87100' #Lee
NOTIFICATION_RECIPIENT_ID = '6047c2e771dcbb0018436f39' #NATE

def getDevices(account):
    # Example of filtering devies by Tag.
    # You can filter by: name, last_input, last_output, bucket, etc.
    print('getting devices')

    # Searching all devices with tag we want
    devices = account.devices.list(1, ['id', 'tags'], amount=10000)
    if devices['status'] is True:
        return devices['result']
    else:
        print(devices['message']) # Error (if status is False)
        return None


def checkLastOutput(last_output):
    """
    Compares last output to MAX_TIME_SINCE_LAST_MEASUREMENT to determine if 
    the sensor is submitting data as expected or not. 
    """
    time_diff_hours = (datetime.now(timezone('UTC')) - last_output) / timedelta(hours=1)
    return time_diff_hours < MAX_HOURS_SINCE_LAST_MEASUREMENT


def sendNotification(account, device_name, device_id, last_output):
    # Convert last_output to central time
    cst = timezone('America/Chicago')
    last_output_pretty = last_output.astimezone(cst).strftime('%c')
    
    data = {
            'title': 'Device Down Alert',
            'message': f'{device_name} may be down. \n Last output: {last_output_pretty}',
            'buttons': [{
                'label': 'Go to device admin page',
                'url': f'https://admin.tago.io/devices/{device_id}',
                'color': 'red',
            }],
    }
  
    account.run.notificationCreate(NOTIFICATION_RECIPIENT_ID,data)
      
    return 


def checkDevice(device_id, account):  
    # get device token 
    print(f'Checking device {device_id}')
    device_token = account.devices.tokenList(device_id)['result'][0]['token']
  
    # find most recent "deployed" value
    last_deployed_status = Device(device_token).find({'variable':'deployed',
                                                    'query':'last_value'})
  
    if last_deployed_status['status'] is True:
    
        # Check if device is deployed
        try:
            is_deployed = last_deployed_status['result'][0]['value']
        except IndexError:
            print(f'No deployment recorded for Device: {device_id}')
            return 
        
        if is_deployed == 'deployed':

          # Get last_output
            info = account.devices.info(device_id)
            if info['status'] is True:

            # check if last_ouput timestamp is within MAX_TIME_SINCE_LAST_MEASUREMENT
                last_output = parse(info['result']['last_output']) # Parse to datetime
                if checkLastOutput(last_output) is False:
                    print(device_id, ' last output outside of expected window:', last_output, 'sending notification.')
                    sendNotification(account, info['result']['name'], device_id, last_output)
                else: 
                    print(device_id, ' last output within expected window:', last_output)
                    return

            else: # Returns if error
                print(info['message'], ' at get info for ', device_id) # Error (if status is False)
                return
        else: # Returns if is not deployed
            print(device_id, ' not deployed.')
            return 
    else:  # Returns if error
        print(last_deployed_status['message'], ' at last_deployed_status ', device_id) # Error (if status is False)
        return 


# The function myAnalysis will run when you execute your analysis
# def myAnalysis(context, scope):
def run_checker():
    print('start')
    try:
        # reads the value of account_token from the environment variable
#         account_token = list(filter(lambda account_token: account_token['key'] == 'account_token', context.environment))
#         account_token = account_token[0]['value']
        account_token = TAGO_ACCOUNT_TOKEN

        if not account_token:
            return print("Missing account_token Environment Variable.")

        my_account = Account(account_token)

        devices = getDevices(my_account)

        for device in devices:
            checkDevice(device['id'], my_account)
        print('done')
    except Exception as e:
        print(e)
    
    return
