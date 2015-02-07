#!/usr/bin/python

"""
	This script will scrape the r-project.org machine learning selection and format the packages
	in github markdown style for this awesome-machine-learning repo.
"""

from pyquery import PyQuery as pq
import urllib
import codecs

text_file = codecs.open("Packages.txt",encoding='utf-8',mode="w")
d = pq(url='http://cran.r-project.org/web/views/MachineLearning.html',opener=lambda url, **kw: urllib.urlopen(url).read())

for e in d("li").items():
	package_name = e("a").html()
	package_link = e("a")[0].attrib['href']
	if '..' in package_link:
		package_link = package_link.replace("..",'http://cran.r-project.org/web')
		dd = pq(url=package_link,opener=lambda url, **kw: urllib.urlopen(url).read())
		package_description = dd("h2").html()
		text_file.write(" [%s](%s) - %s \n" % (package_name,package_link,package_description))
		# print "* [%s](%s) - %s" % (package_name,package_link,package_description)

	index += 1
