# file: sp500_project/spiders/sp500_perf.py
import scrapy

class Sp500PerfSpider(scrapy.Spider):
    name = "sp500_perf"
    allowed_domains = ["slickcharts.com"]
    start_urls = ["https://www.slickcharts.com/sp500/performance"]

    def parse(self, response):
        """
        Parse the S&P 500 performance page and yield one item per row:
        fields: rank, company, symbol, ytd_return
        """

        # The table rows for the performance list are in the page's main table.
        # We select the rows and iterate.
        rows = response.css("table.table tbody tr")

        for row in rows:
            # There are usually four columns: Rank, Company, Symbol, YTD Return
            # Adjust the nth-child selectors if the page HTML changes.
            rank = row.css("td:nth-child(1)::text").get()
            company = row.css("td:nth-child(2) a::text").get()
            symbol = row.css("td:nth-child(3)::text").get()
            ytd_return = row.css("td:nth-child(4)::text").get()

            # Clean results (strip whitespace)
            rank = rank.strip() if rank else None
            company = company.strip() if company else None
            symbol = symbol.strip() if symbol else None
            ytd_return = ytd_return.strip() if ytd_return else None

            yield {
                "rank": rank,
                "company": company,
                "symbol": symbol,
                "ytd_return": ytd_return
            }