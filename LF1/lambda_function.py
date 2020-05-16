import json
import boto3
import email
from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences

vocabulary_length = 9013

def lambda_handler(event, context):
    email_msg = extract_email(event)
    body = email_msg.get_payload()
    if isinstance(body, list):
        body = body[0].get_payload()
    email_body = body.rstrip()
    
    label, conf = predict_spam(email_body)
    
    msg = '''We received your email sent at %s with the subject %s.
Here is a 240 character sample of the email body: 
%s
The email was categorized as %s with a %f%% confidence.''' % (email_msg['Date'], email_msg['Subject'], email_body[:240], label, conf*100)
    
    print(msg)
    
    if '@assign4.tech' in email_msg['From'] or '@assign4.me' in email_msg['From']:
        return msg
    
    sendEmail(email_msg['From'], email_msg['To'], msg)
    return msg


def extract_email(event):
    bucket = event['Records'][0]['s3']['bucket']['name']
    filename = event['Records'][0]['s3']['object']['key']
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, filename)
    body = obj.get()['Body'].read()
    msg = email.message_from_bytes(body)
    return msg
    
def predict_spam(text):
    # grab environment variables
    ENDPOINT_NAME = "sms-spam-classifier-mxnet-2020-05-12-19-24-04-655"
    runtime= boto3.client('runtime.sagemaker')
    
    one_hot_test_messages = one_hot_encode([text], vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=json.dumps(encoded_test_messages))
    # print(response)
    # result = response['Body'].read().decode()
    result = json.loads(response['Body'].read().decode())
    # print(result)
    pred = int(result['predicted_label'][0][0])
    prob = float(result['predicted_probability'][0][0])
    # pred = int(result['predictions'][0]['score'])
    # predicted_label = 'SPAM' if pred == 1 else 'HAM'
    label = 'SPAM' if pred == 1 else 'HAM'
    conf = prob if pred == 1 else 1-prob
    print("label: %f \n prob: %f" % (pred, prob))
    return label, conf

def sendEmail(fromAddr, toAddr, msg):
    client = boto3.client('ses')
    response = client.send_email(
    Destination={
        'ToAddresses': [fromAddr],
    },
    Message={
        'Body': {
            'Text': {
                'Charset': 'UTF-8',
                'Data': msg,
            },
        },
        'Subject': {
            'Charset': 'UTF-8',
            'Data': 'Reply For Classification result',
        },
    },
    Source=toAddr,
    )
