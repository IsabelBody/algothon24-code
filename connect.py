import boto3
import paramiko

# AWS credentials and region configuration
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'
region_name = 'ap-southeast-2'

# Connect to AWS services
ec2_client = boto3.client('ec2', region_name=region_name)

# Define EC2 instance parameters
instance_type = 't2.micro'  # Choose instance type based on your computation needs
ami_id = 'your_ami_id'  # AMI with Python and necessary libraries pre-installed

# Start an EC2 instance
response = ec2_client.run_instances(
    ImageId=ami_id,
    InstanceType=instance_type,
    KeyName='algokey',
    MinCount=1,
    MaxCount=1
)

instance_id = response['Instances'][0]['InstanceId']

# Wait for instance to be running
waiter = ec2_client.get_waiter('instance_running')
waiter.wait(InstanceIds=[instance_id])

# Connect to the instance using SSH (assuming you have SSH access set up)
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname='ec2-instance-public-dns', username='your_username', key_filename='path_to_your_private_key')

# Execute your Python script remotely
stdin, stdout, stderr = ssh_client.exec_command('python3 /path/to/your_script.py')

# Read output or wait for completion, handle errors, etc.

# Close SSH connection
ssh_client.close()

# Terminate EC2 instance after use (optional)
ec2_client.terminate_instances(InstanceIds=[instance_id])