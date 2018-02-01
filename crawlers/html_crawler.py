from urllib.request import urlopen

from bs4 import BeautifulSoup

class html_crawler():

    @staticmethod
    def html_to_text(link, class_list):
        html_file = urlopen(link)

        texts = {}

        soup = BeautifulSoup(html_file.read(), 'html.parser')

        for class_name in class_list:

            text = ''

            for tag in soup.find_all(class_=class_name):
                text += str(tag.get_text())
                text += ' '

            texts[class_name] = text

        return texts


if __name__ == '__main__':

    textual_data = html_crawler.html_to_text(
        link='https://www.theguardian.com/music/2018/feb/01/bryan-ferry-roxy-music-invented-new-pop-game-for-anything',
        class_list=['content__headline', 'content__standfirst', 'content__article-body'])

    import re

    for key in textual_data.keys():
        print(key)
        print(re.sub('\n\n+', '\n', textual_data[key]))