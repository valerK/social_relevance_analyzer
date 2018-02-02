import json, csv
import time
import urllib.request as Request
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class fb_crawler():

    def __init__(self, app_id, app_secret):

        self.app_id = app_id
        self.app_secret = app_secret

    def authenticate(self):
        access_token = Request.urlopen(url='https://graph.facebook.com/v2.10/oauth/access_token?client_id=' +
                                           self.app_id +
                                           '&client_secret=' +
                                           self.app_secret +
                                           '&grant_type=client_credentials')

        self.access_token = json.loads(access_token.read().decode())['access_token']

    def writing_post_csv(self, writer, page, posts):
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

    def crawl_posts(self, page_list, posts_file_path):

        self.authenticate()

        with open(file=posts_file_path, mode='a') as output_csv:
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

                self.writing_post_csv(writer, page, answer['posts']['data'])

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

                    self.writing_post_csv(writer, page, answer['data'])

                    try:
                        logging.info(page + ' : ' + answer['data'][-1]['created_time'])
                    except IndexError:
                        pass

                    try:
                        next_url = answer['paging']['next']
                    except KeyError:

                        logging.info('No more posts for page ' + page)

                        break

    def crawl_comments(self, post_id, writer):

        while True:
            try:
                answer = json.loads(Request.urlopen('https://graph.facebook.com/v2.10/' + post_id +
                                                    '?fields=comments{message,created_time}'
                                                    '&access_token=' + self.access_token).read().decode())

                break

            except Exception as e:

                logging.info('An error happpened while downloading comments. Waiting 30s before trying again...')
                logging.info(str(e))

                time.sleep(30)

                continue

        for comment in answer['comments']['data']:
            writer.writerow([post_id, comment['id'], comment['created_time'], comment['message']])

        try:
            logging.info('For post ' + post_id + ' last comment created at ' +
                         answer['comments']['data'][-1]['created_time'] + ' has been downloaded.')
        except KeyError:

            logging.info('There were no comments for this post.')

            return

        try:
            next_url = answer['comments']['paging']['next']
        except KeyError:

            logging.info('No more comments for post ' + post_id)

            return

        j = 0

        while True:
            try:
                answer = json.loads(Request.urlopen(next_url).read().decode())

            except Exception as e:

                logging.info('An error happpened while downloading comments. Waiting 30s before trying again...')
                logging.info(str(e))

                time.sleep(30)

                j += 1

                if j > 5:

                    break

                continue

            for comment in answer['data']:
                writer.writerow([post_id, comment['id'], comment['created_time'], comment['message']])

            logging.info('For post ' + post_id + ' last comment created at ' +
                         answer['data'][-1]['created_time'] + ' has been downloaded.')

            try:
                next_url = answer['paging']['next']

                j = 0
            except KeyError:

                logging.info('No more comments for post ' + post_id)

                break

    def crawl_comments_from_post_file(self, posts_file_path, comments_file_path):

        self.authenticate()

        with open(file=posts_file_path, mode='r') as input_csv:

            reader = csv.DictReader(input_csv)

            with open(file=comments_file_path, mode='w') as output_csv:
                writer = csv.writer(output_csv)
                writer.writerow(['post_id', 'id', 'created_time', 'message'])

                for row in reader:
                    self.crawl_comments(row['id'], writer)


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

    crawler =  fb_crawler(app_id='XXXXXXXXXXXXXXX', # Facebook app_id
                          app_secret='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') # Facebook app_secret

    crawler.crawl_posts(page_list=page_list, posts_file_path='./posts.csv')

    crawler.crawl_comments_from_post_file(posts_file_path='./posts.csv',
                                          comments_file_path='./comments.csv')