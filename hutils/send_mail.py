import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from email.mime.image import MIMEImage
import os

#设置登录及服务器信息
mail_host = 'smtp.163.com'
mail_user = 'hengguoee'
mail_pass = 'HURFTKVQUYWFROIX'
sender = 'hengguoee@163.com'
receivers = ['1134537617@qq.com']


def init_mailserver():
    try:
        smtpObj = smtplib.SMTP()
        # 连接到服务器
        smtpObj.connect(mail_host, 25)
        # 登录到服务器
        smtpObj.login(mail_user, mail_pass)
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误

    return smtpObj

def send_mail(smtpObj, content, title, file_path):
    message = MIMEMultipart()
    part_text = MIMEText(content)
    message.attach(part_text)  # 把正文加到邮件体里面去
    message['Subject'] = title
    message['From'] = sender
    message['To'] = receivers[0]

    if os.path.exists(file_path):
        part_attach1 = MIMEApplication(open(file_path, 'rb').read())  # 打开附件
        part_attach1.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file_path))  # 为附件命名
        message.attach(part_attach1)

    try:
        smtpObj.sendmail(
            sender, receivers, message.as_string())

    except smtplib.SMTPException as e:
        print('error', e)

    print('successfully send mail')


# smtpObj.quit()

if __name__ == '__main__':
    mail_handler = init_mailserver()
    send_mail(mail_handler, 'test', 'append file', r'F:\Project\UncalibratedPS\utils\fileio.py')