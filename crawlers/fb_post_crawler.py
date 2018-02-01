import json, csv
import time
import urllib.request as Request
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class fb_crawler():

    def __init__(self, app_id, app_secret, file_path):

        self.app_id = app_id
        self.app_secret = app_secret

        access_token = Request.urlopen(url='https://graph.facebook.com/v2.10/oauth/access_token?client_id=' +
                                           app_id +
                                           '&client_secret=' +
                                           app_secret +
                                           '&grant_type=client_credentials')

        self.access_token = json.loads(access_token.read().decode())['access_token']

        self.file_path = file_path

    def writing_csv(self, writer, page, posts):
        for post in posts:

            try:
                desc = post['description']
            except KeyError:
                desc = ''

            try:
                mess = post['message']
            except KeyError:
                mess = ''

            try:
                link = post['link']
            except KeyError:
                link = ''

            writer.writerow([
                page, post['id'], mess, desc, link, post['created_time'],
                post['like']['summary']['total_count'], post['love']['summary']['total_count'],
                post['haha']['summary']['total_count'], post['wow']['summary']['total_count'],
                post['angry']['summary']['total_count'], post['sad']['summary']['total_count']])

    def crawl_posts(self, page_list):
        with open(file=self.file_path, mode='a') as output_csv:
            writer = csv.writer(output_csv)
            writer.writerow(['page', 'id', 'message', 'description', 'link', 'created_time',
                             'like', 'love', 'haha', 'wow', 'angry', 'sad'])

            for page in page_list:

                while True:
                    try:
                        answer = json.loads(Request.urlopen('https://graph.facebook.com/v2.10/' + page +
                                                            '?fields=posts{id,message,description,link,created_time,'
                                                            'reactions.type(LIKE).limit(0).summary(true).as(like),'
                                                            'reactions.type(LOVE).limit(0).summary(true).as(love),'
                                                            'reactions.type(HAHA).limit(0).summary(true).as(haha),'
                                                            'reactions.type(WOW).limit(0).summary(true).as(wow),'
                                                            'reactions.type(ANGRY).limit(0).summary(true).as(angry),'
                                                            'reactions.type(SAD).limit(0).summary(true).as(sad)}'
                                                            '&access_token=' + self.access_token).read().decode())

                        break

                    except Exception as e:

                        time.sleep(30)

                        logging.info('An error happpened while downloading posts. Waiting 30s before trying again...')
                        logging.info(str(e))


                        continue

                self.writing_csv(writer, page, answer['posts']['data'])

                logging.info(page + ' : ' + answer['posts']['data'][-1]['created_time'])

                try:
                    next_url = answer['posts']['paging']['next']
                except KeyError:

                    logging.info('No more posts for page ' + page)

                    continue

                while True:
                    try:
                        answer = json.loads(Request.urlopen(next_url).read().decode())
                    except Exception as e:
                        time.sleep(30)

                        logging.info('An error happpened while downloading posts. Waiting 30s before trying again...')
                        logging.info(str(e))

                        continue

                    self.writing_csv(writer, page, answer['data'])

                    try:
                        logging.info(page + ' : ' + answer['data'][-1]['created_time'])
                    except IndexError:
                        pass

                    try:
                        next_url = answer['paging']['next']
                    except KeyError:

                        logging.info('No more posts for page ' + page)

                        break


if __name__ == '__main__':

    page_list = [
        'theguardian',
        'washingtonpost',
        'nytimes',
        'wsj',
        'financialtimes',
        'timesandsundaytimes',
        'newyorker',
        'TheAtlantic',
        'TheEconomist',
        'foreign.policy.magazine',
        'politico',
        'POLITICOeu',
        'cnn',
        'bbcnews',
        'ABCNews'
    ]

    fb_crawler(app_id='XXXXXXXXXXXXXXX', # Facebook app_id
               app_secret='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', # Facebook app_secret
               file_path='./posts.csv').crawl_posts(page_list)
