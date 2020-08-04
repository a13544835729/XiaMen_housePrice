import requests
from lxml import etree
import time
import random
from fake_useragent import UserAgent
import pymysql
from tqdm import tqdm

class LianjiaSpider(object):
    def __init__(self):
        #一级页面连接
        self.one_url = 'https://xm.lianjia.com/ershoufang/pg{}a{}/'
        #请求头
        self.headers={ 'User-Agent':UserAgent().random }
        #连接数据库
        self.db = pymysql.connect(
            host='localhost', user='root', password='123456', database='lianjiaData', charset='utf8', port=3306
        )
        self.cur= self.db.cursor()

    #解析一级页面
    def parse_html(self,url):
        # 有问题页面,尝试3次,如果不行直接抓取下一页数据
        for i in range(2):
            try:
                html = requests.get(url=url,headers=self.headers,timeout=3).content.decode('utf-8','ignore')
                self.get_data(html)
                break
            except Exception as e:
                print('Retry')



    #抓取一级页面数据
    def get_data(self,html):
        p = etree.HTML(html)
        # 基准xpath: [<element li at xxx>,<element li>]
        li_list = p.xpath('//ul[@class="sellListContent"]/li[@class="clear LOGVIEWDATA LOGCLICKDATA"]')
        # for遍历,依次提取每个房源信息,放到字典item中
        item = {}
        for li in li_list:
            # 名称+区域
            name_list = li.xpath('.//div[@class="positionInfo"]/a[1]/text()')
            item['name'] = name_list[0].strip() if name_list else None
            address_list = li.xpath('.//div[@class="positionInfo"]/a[2]/text()')
            item['address'] = address_list[0].strip() if address_list else None
            # 户型+面积+方位+是否精装+楼层+年代+类型
            # h_list: ['']
            h_list = li.xpath('.//div[@class="houseInfo"]/text()')
            if h_list:
                info_list = h_list[0].split('|')
                if len(info_list) == 7:
                    item['model'] = info_list[0].strip()
                    item['area'] = info_list[1].strip()
                    item['direct'] = info_list[2].strip()
                    item['perfect'] = info_list[3].strip()
                    item['floor'] = info_list[4].strip()
                    item['year'] = info_list[5].strip()[:-2]
                    item['type'] = info_list[6].strip()
                else:
                    item['model'] = item['area'] = item['direct'] = item['perfect'] = item['floor'] = item['year'] = item['type'] = None
            else:
                item['model'] = item['area'] = item['direct'] = item['perfect'] = item['floor'] = item['year'] = item['type'] = None

            # 总价+单价
            total_list = li.xpath('.//div[@class="totalPrice"]/span/text()')
            item['total'] = total_list[0].strip() if total_list else None
            unit_list = li.xpath('.//div[@class="unitPrice"]/span/text()')
            item['unit'] = unit_list[0].strip() if unit_list else None

            #二级页面url列表
            two_url_list=li.xpath('.//div[@class="info clear"]//div[@class="title"]//a/@href')
            two_url=two_url_list[0].strip() if two_url_list else None
            #地铁交通
            traffic=self.parse_two_html(two_url)
            # print(item)
            #保存到数据库
            self.save_mysql(item['name'],item['address'],item['model'],item['area'],item['direct'],item['perfect'],item['floor'],item['year'],item['type'],item['total'],item['unit'],traffic)


    #解析二级页面
    def parse_two_html(self,url):
        # 有问题页面,尝试3次,如果不行直接抓取下一页数据
        for i in range(2):
            try:
                html = requests.get(url=url, headers=self.headers, timeout=3).content.decode('utf-8', 'ignore')
                k = etree.HTML(html)
                sub=k.xpath('.//div[@class="aroundInfo"]//div[@class="areaName"]/a/text()')
                sub=sub[0].strip() if sub else None
                return sub
            except Exception as e:
                print('Retry')


    #获取页面数
    def get_page(self,url):
        # 有问题页面,尝试3次,如果不行直接抓取下一页数据
        for i in range(2):
            try:
                html = requests.get(url=url, headers=self.headers, timeout=3).content.decode('utf-8', 'ignore')
                g = etree.HTML(html)
                pageNum = g.xpath('//div[@class="page-box fr"]/div[1]/@page-data')
                pageNum = eval(pageNum[0].strip())['totalPage']
                return pageNum
            except Exception as e:
                print('Retry')


    #保存到数据库
    def save_mysql(self,name,address,model,area,direct,perfect,floor,year,type,total,unit,traffic):
        sql='insert into housePrice (name,address,model,area,direct,perfect,floor,year,type,total,unit,traffic) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        try:
            # 检查连接是否断开
            self.db.ping(reconnect=True)
            self.cur.execute(sql,[name,address,model,area,direct,perfect,floor,year,type,total,unit,traffic])
            self.db.commit()
        except:
            self.db.rollback()




    def run(self):
        print('开始抓取')
        for i in range(1,9):
            page_url=self.one_url.format(1,i)
            print(page_url)
            pageNum=self.get_page(page_url)
            print('第{}个区域抓取'.format(i))
            for pg in tqdm(range(1,int(pageNum)+1)):
                try:
                    one_url = self.one_url.format(pg,i)
                    self.parse_html(one_url)
                    time.sleep(random.randint(1))
                except Exception as e:
                    print(e)
                    continue
        print('END')


if __name__ == '__main__':
    spider = LianjiaSpider()
    spider.run()













