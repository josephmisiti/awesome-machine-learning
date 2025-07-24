#!/usr/bin/python

"""
    This script will scrape the r-project.org machine learning selection and
    format the packages in github markdown style for this
    awesome-machine-learning repo.
"""

import requests
from bs4 import BeautifulSoup

with open("Packages.txt", "w", encoding="utf-8") as text_file:
    base_url = "https://cran.r-project.org/web/views/MachineLearning.html"
    res = requests.get(base_url)
    soup = BeautifulSoup(res.text, "html.parser")

    for li in soup.select("li"):
        a = li.find("a")
        if not a:
            continue
        package_name = a.text
        package_link = a.get("href")

        if ".." in package_link:
            package_link = package_link.replace("..", "https://cran.r-project.org/web")
            try:
                package_res = requests.get(package_link)
                package_soup = BeautifulSoup(package_res.text, "html.parser")
                description_tag = package_soup.find("h2")
                package_description = description_tag.text if description_tag else "No description"
            except Exception as e:
                package_description = "Error fetching description"

            text_file.write(f"[{package_name}]({package_link}) - {package_description}\n")