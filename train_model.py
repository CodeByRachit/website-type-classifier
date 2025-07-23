# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests
from bs4 import BeautifulSoup
import time
import random
import re # Added for URL cleaning

print("--- Starting Model Training Script ---")

# --- 1. Prepare your Training Data ---
# IMPORTANT: For a truly robust model, you'll need significantly more diverse
# and larger datasets. This is a working example.
# Each category MUST have at least 2 unique, scrapeable URLs.
# I've increased it to 3 per category for better robustness in splitting.
training_data_raw = [
    # E-commerce (expanded)
    {"url": "https://www.ebay.com/b/electronics/bn7000259646", "type": "e-commerce"},
    {"url": "https://www.walmart.com/cp/electronics/3944", "type": "e-commerce"},
    {"url": "https://www.zappos.com/c/shoes", "type": "e-commerce"},
    {"url": "https://www.asos.com/us/", "type": "e-commerce"},
    {"url": "https://www.nordstrom.com/browse", "type": "e-commerce"},
    {"url": "https://www.shein.com/", "type": "e-commerce"},
    {"url": "https://www.temu.com/", "type": "e-commerce"},
    {"url": "https://www.etsy.com/c/home-and-living", "type": "e-commerce"}, # Often blocks, but try
    {"url": "https://www.homedepot.com/b/Appliances/N-5yc1vZc3ny", "type": "e-commerce"},
    
    {"url": "https://www.target.com/c/electronics/-/N-5xtg6", "type": "e-commerce"},
    {"url": "https://www.bestbuy.com/site/electronics/all-electronics/pcmcat242800050001.c", "type": "e-commerce"},
    {"url": "https://www.ikea.com/us/en/cat/furniture-fu003/", "type": "e-commerce"},
    {"url": "https://www.alibaba.com/products", "type": "e-commerce"},
    {"url": "https://www.amazon.com/Best-Sellers/zgbs", "type": "e-commerce"},
    {"url": "https://www.aliexpress.com/category/100003070/electronics.html", "type": "e-commerce"},

    # News (expanded)
    {"url": "https://www.nytimes.com/section/world", "type": "news"},
    {"url": "https://www.bbc.com/news/world", "type": "news"},
    {"url": "https://www.reuters.com/world", "type": "news"},
    {"url": "https://www.cnn.com/world", "type": "news"},
    {"url": "https://www.foxnews.com/world", "type": "news"},
    {"url": "https://www.washingtonpost.com/world/", "type": "news"},
    {"url": "https://www.apnews.com/hub/world-news", "type": "news"},
    {"url": "https://www.bloomberg.com/news", "type": "news"},
    {"url": "https://www.wsj.com/news/world", "type": "news"},
    {"url": "https://www.usatoday.com/news/world/", "type": "news"},
    {"url": "https://www.cbsnews.com/world/", "type": "news"},
    {"url": "https://www.abcnews.go.com/International/", "type": "news"},
    {"url": "https://www.nbcnews.com/world", "type": "news"},
    {"url": "https://www.theguardian.com/world", "type": "news"},
    {"url": "https://www.independent.co.uk/world", "type": "news"},

    # Blog (expanded)
    {"url": "https://www.techcrunch.com/startups/", "type": "blog"},
    {"url": "https://www.huffpost.com/entry/blog", "type": "blog"},
    {"url": "https://www.medium.com/topic/technology", "type": "blog"},
    {"url": "https://www.theverge.com/tech", "type": "blog"},
    {"url": "https://www.engadget.com/topics/tech/", "type": "blog"},
    {"url": "https://lifehacker.com/tech", "type": "blog"},
    {"url": "https://www.cnet.com/tech/", "type": "blog"},
    {"url": "https://www.slashdot.org/", "type": "blog"},
    {"url": "https://www.boingboing.net/", "type": "blog"},
    {"url": "https://www.lifehack.org/", "type": "blog"},
    {"url": "https://gizmodo.com/technology", "type": "blog"},
    {"url": "https://www.wired.com/tag/blogs/", "type": "blog"},
    {"url": "https://www.howtogeek.com/", "type": "blog"},
    {"url": "https://www.digitaltrends.com/computing/", "type": "blog"},
    {"url": "https://www.mashable.com/tech", "type": "blog"},

    # Job board (expanded)
    {"url": "https://www.dice.com/jobs", "type": "job board"},
    {"url": "https://www.simplyhired.com/search", "type": "job board"},
    {"url": "https://www.theladders.com/jobs", "type": "job board"},
    {"url": "https://www.jobs2careers.com/", "type": "job board"},
    {"url": "https://www.jobindex.dk/", "type": "job board"},
    {"url": "https://www.seek.com.au/", "type": "job board"},
    {"url": "https://www.naukri.com/", "type": "job board"},
    {"url": "https://www.flexjobs.com/", "type": "job board"},
    {"url": "https://www.weworkremotely.com/", "type": "job board"},
    {"url": "https://www.careerbuilder.com/jobs", "type": "job board"},
    {"url": "https://www.monster.com/jobs/", "type": "job board"},
    {"url": "https://www.indeed.com/jobs", "type": "job board"}, # Can be tricky to scrape
    {"url": "https://www.glassdoor.com/jobs/", "type": "job board"},
    {"url": "https://www.upwork.com/freelance-jobs/", "type": "job board"},
    {"url": "https://www.ziprecruiter.com/jobs/", "type": "job board"},

    # Health (expanded)
    {"url": "https://www.nhs.uk/conditions/", "type": "health"},
    {"url": "https://www.everydayhealth.com/conditions/", "type": "health"},
    {"url": "https://www.health.harvard.edu/diseases-and-conditions", "type": "health"},
    {"url": "https://www.medicinenet.com/diseases_and_conditions/article.htm", "type": "health"},
    {"url": "https://www.kidshealth.org/en/parents/medical/", "type": "health"},
    {"url": "https://www.clevelandclinic.org/health", "type": "health"},
    {"url": "https://www.drugs.com/condition/", "type": "health"},
    {"url": "https://www.verywellhealth.com/conditions", "type": "health"},
    {"url": "https://www.hopkinsmedicine.org/health/", "type": "health"},
    {"url": "https://www.nimh.nih.gov/health/topics", "type": "health"},
    {"url": "https://www.mayoclinic.org/diseases-conditions", "type": "health"},
    {"url": "https://www.webmd.com/diseases/a-z-list", "type": "health"},
    {"url": "https://www.medlineplus.gov/diseases.html", "type": "health"},
    {"url": "https://www.who.int/health-topics", "type": "health"},
    {"url": "https://www.cdc.gov/diseasesconditions/index.html", "type": "health"},

    # Wiki (expanded)
    {"url": "https://www.mediawiki.org/wiki/Main_Page", "type": "wiki"},
    {"url": "https://www.wiktionary.org/", "type": "wiki"},
    {"url": "https://www.wikiversity.org/", "type": "wiki"},
    {"url": "https://www.wikiquote.org/", "type": "wiki"},
    {"url": "https://www.wikibooks.org/", "type": "wiki"},
    {"url": "https://www.scholarpedia.org/", "type": "wiki"},
    {"url": "https://www.sourcewatch.org/", "type": "wiki"},
    {"url": "https://www.rationalwiki.org/wiki/Main_Page", "type": "wiki"},
    {"url": "https://www.wikiprojectmed.org/", "type": "wiki"},
    {"url": "https://www.lyricwiki.org/", "type": "wiki"},
    {"url": "https://www.wikipedia.org/", "type": "wiki"},
    {"url": "https://www.wikivoyage.org/", "type": "wiki"},
    {"url": "https://www.wikinews.org/", "type": "wiki"},
    {"url": "https://www.wikisource.org/", "type": "wiki"},
    {"url": "https://www.wikispecies.org/", "type": "wiki"},

    # Forum (expanded)
    {"url": "https://forums.digitalpoint.com/", "type": "forum"},
    {"url": "https://www.warriorforum.com/", "type": "forum"},
    {"url": "https://www.webmasterworld.com/", "type": "forum"},
    {"url": "https://www.sitepoint.com/community/", "type": "forum"},
    {"url": "https://forums.anandtech.com/", "type": "forum"},
    {"url": "https://www.cnet.com/forums/", "type": "forum"},
    {"url": "https://www.tomshardware.com/forums/", "type": "forum"},
    {"url": "https://www.dpreview.com/forums/", "type": "forum"},
    {"url": "https://www.bleepingcomputer.com/forums/", "type": "forum"},
    {"url": "https://www.hackforums.net/", "type": "forum"},
    {"url": "https://www.forums.macrumors.com/", "type": "forum"},
    {"url": "https://www.gaming.stackexchange.com/", "type": "forum"}, # StackExchange is a Q&A forum format
    {"url": "https://www.stackoverflow.com/", "type": "forum"},
    {"url": "https://www.ubuntuforums.org/", "type": "forum"},
    {"url": "https://www.rpg.net/forums/", "type": "forum"},

    # Travel (expanded)
    {"url": "https://www.travelocity.com/Destinations", "type": "travel"},
    {"url": "https://www.priceline.com/vacation-packages", "type": "travel"},
    {"url": "https://www.roughguides.com/destinations/", "type": "travel"},
    {"url": "https://www.fodors.com/destinations", "type": "travel"},
    {"url": "https://www.frommers.com/destinations", "type": "travel"},
    {"url": "https://www.orbitz.com/Destinations", "type": "travel"},
    {"url": "https://www.skyscanner.com/", "type": "travel"},
    {"url": "https://www.cntraveler.com/destinations", "type": "travel"},
    {"url": "https://www.atlasobscura.com/", "type": "travel"},
    {"url": "https://www.ricksteves.com/europe", "type": "travel"},
    {"url": "https://www.tripadvisor.com/Attractions", "type": "travel"},
    {"url": "https://www.lonelyplanet.com/destinations", "type": "travel"},
    {"url": "https://www.expedia.com/Destinations", "type": "travel"},
    {"url": "https://www.kayak.com/flights", "type": "travel"},
    {"url": "https://www.booking.com/index.html", "type": "travel"},

    # Educational (expanded)
    {"url": "https://www.pluralsight.com/courses", "type": "educational"},
    {"url": "https://www.skillshare.com/browse", "type": "educational"},
    {"url": "https://www.lynda.com/learning-paths", "type": "educational"}, # Now LinkedIn Learning
    {"url": "https://www.masterclass.com/classes", "type": "educational"},
    {"url": "https://www.open.ac.uk/courses", "type": "educational"},
    {"url": "https://www.alison.com/courses", "type": "educational"},
    {"url": "https://www.udacity.com/courses/all", "type": "educational"},
    {"url": "https://www.mit.edu/education/", "type": "educational"},
    {"url": "https://www.openculture.com/freeonlinecourses", "type": "educational"},
    {"url": "https://www.classcentral.com/", "type": "educational"},
    {"url": "https://www.coursera.org/browse", "type": "educational"},
    {"url": "https://www.edx.org/course", "type": "educational"},
    {"url": "https://www.khanacademy.org/", "type": "educational"},
    {"url": "https://www.nptel.ac.in/courses", "type": "educational"}, # Indian government initiative
    {"url": "https://www.byjus.com/", "type": "educational"}, # Ed-tech for school
    {"url": "https://www.geeksforgeeks.org/", "type": "educational"}, # Programming education

    # Corporate (expanded)
    {"url": "https://www.oracle.com/corporate/", "type": "corporate"},
    {"url": "https://www.salesforce.com/company/", "type": "corporate"},
    {"url": "https://www.dell.com/en-us/dt/corporate/index.htm", "type": "corporate"},
    {"url": "https://www.hp.com/us-en/hp-information.html", "type": "corporate"},
    {"url": "https://www.samsung.com/us/about-us/", "type": "corporate"},
    {"url": "https://www.intel.com/content/www/us/en/about-intel.html", "type": "corporate"},
    {"url": "https://www.cisco.com/c/en/us/about.html", "type": "corporate"},
    {"url": "https://www.accenture.com/us-en/about", "type": "corporate"},
    {"url": "https://www.pwc.com/gx/en/about.html", "type": "corporate"},
    {"url": "https://www.nike.com/about", "type": "corporate"},
    {"url": "https://www.microsoft.com/en-us/about", "type": "corporate"},
    {"url": "https://www.apple.com/about/", "type": "corporate"},
    {"url": "https://www.google.com/about/", "type": "corporate"},
    {"url": "https://www.ibm.com/about", "type": "corporate"},
    {"url": "https://www.meta.com/about/", "type": "corporate"},

    # Government (expanded)
    {"url": "https://www.france.fr/en", "type": "government"},
    {"url": "https://www.india.gov.in/", "type": "government"},
    {"url": "https://www.gov.br/en", "type": "government"},
    {"url": "https://www.gov.za/", "type": "government"},
    {"url": "https://www.singapore.gov.sg/", "type": "government"},
    {"url": "https://www.irs.gov/", "type": "government"},
    {"url": "https://www.nsw.gov.au/", "type": "government"},
    {"url": "https://www.servicecanada.gc.ca/", "type": "government"},
    {"url": "https://www.gov.ie/en/", "type": "government"},
    {"url": "https://www.mygov.in/", "type": "government"},
    {"url": "https://www.usa.gov/", "type": "government"},
    {"url": "https://www.gov.uk/", "type": "government"},
    {"url": "https://www.bundesregierung.de/breg-de/english", "type": "government"},
    {"url": "https://www.australia.gov.au/", "type": "government"},
    {"url": "https://www.whitehouse.gov/", "type": "government"},

    # Non-profit (expanded)
    {"url": "https://www.oxfam.org/en/donate", "type": "non-profit"},
    {"url": "https://www.savethechildren.org/us/donate", "type": "non-profit"},
    {"url": "https://www.habitat.org/donate/", "type": "non-profit"},
    {"url": "https://www.wfp.org/donate", "type": "non-profit"},
    {"url": "https://www.care.org/donate/", "type": "non-profit"},
    {"url": "https://www.mercycorps.org/donate", "type": "non-profit"},
    {"url": "https://www.conservation.org/donate", "type": "non-profit"},
    {"url": "https://www.directrelief.org/donate/", "type": "non-profit"},
    {"url": "https://www.charitywater.org/donate", "type": "non-profit"},
    {"url": "https://www.stjude.org/donate", "type": "non-profit"},
    {"url": "https://www.redcross.org/donations/ways-to-donate.html", "type": "non-profit"},
    {"url": "https://www.doctorswithoutborders.org/", "type": "non-profit"},
    {"url": "https://www.amnesty.org/en/donate/", "type": "non-profit"},
    {"url": "https://www.unicef.org/donate", "type": "non-profit"},
    {"url": "https://www.worldwildlife.org/", "type": "non-profit"},

    # Social media (expanded)
    {"url": "https://www.linkedin.com/feed/", "type": "social media"},
    {"url": "https://www.reddit.com/explore", "type": "social media"},
    {"url": "https://www.tumblr.com/explore/trending", "type": "social media"},
    {"url": "https://www.discord.com/channels/explore", "type": "social media"},
    {"url": "https://www.mastodon.social/explore", "type": "social media"},
    {"url": "https://www.whatsapp.com/channels", "type": "social media"},
    {"url": "https://www.weibo.com/", "type": "social media"},
    {"url": "https://www.nextdoor.com/", "type": "social media"},
    {"url": "https://www.vk.com/", "type": "social media"},
    {"url": "https://www.meetup.com/", "type": "social media"},
    {"url": "https://www.facebook.com/", "type": "social media"}, # Can be tricky to scrape for public content
    {"url": "https://www.twitter.com/", "type": "social media"}, # Now X.com, hard to scrape without API
    {"url": "https://www.instagram.com/", "type": "social media"}, # Very hard to scrape without API
    {"url": "https://www.pinterest.com/ideas/", "type": "social media"},
    {"url": "https://www.snapchat.com/", "type": "social media"},

    # Video streaming (expanded)
    {"url": "https://www.disneyplus.com/home", "type": "video streaming"},
    {"url": "https://www.primevideo.com/storefront", "type": "video streaming"},
    {"url": "https://www.peacocktv.com/stream", "type": "video streaming"},
    {"url": "https://www.vevo.com/", "type": "video streaming"},
    {"url": "https://www.dailymotion.com/", "type": "video streaming"},
    {"url": "https://www.hbomax.com/", "type": "video streaming"},
    {"url": "https://www.paramountplus.com/home/", "type": "video streaming"},
    {"url": "https://www.apple.com/apple-tv-plus/", "type": "video streaming"},
    {"url": "https://www.roku.com/whats-on", "type": "video streaming"},
    {"url": "https://www.youtube.com/", "type": "video streaming"},
    {"url": "https://www.twitch.tv/directory/all", "type": "video streaming"},
    {"url": "https://www.netflix.com/browse", "type": "video streaming"},
    {"url": "https://www.hulu.com/hub/browse", "type": "video streaming"},
    {"url": "https://www.vimeo.com/categories", "type": "video streaming"},
    {"url": "https://www.crunchyroll.com/videos/anime", "type": "video streaming"},

    # Gaming (expanded)
    {"url": "https://www.miniclip.com/games", "type": "gaming"},
    {"url": "https://www.addictinggames.com/", "type": "gaming"},
    {"url": "https://www.poki.com/", "type": "gaming"},
    {"url": "https://www.crazygames.com/", "type": "gaming"},
    {"url": "https://www.newgrounds.com/games", "type": "gaming"},
    {"url": "https://www.gamespot.com/games/", "type": "gaming"},
    {"url": "https://www.ign.com/games", "type": "gaming"},
    {"url": "https://www.roblox.com/discover", "type": "gaming"},
    {"url": "https://www.gamejolt.com/games", "type": "gaming"},
    {"url": "https://www.friv.com/", "type": "gaming"},
    {"url": "https://store.steampowered.com/genre/Free%20to%20Play/", "type": "gaming"}, # Steam free to play
    {"url": "https://www.epicgames.com/store/en-US/browse", "type": "gaming"},
    {"url": "https://www.nintendo.com/games/", "type": "gaming"},
    {"url": "https://www.playstation.com/en-us/games/", "type": "gaming"},
    {"url": "https://www.xbox.com/en-US/games", "type": "gaming"},

    # Event (expanded)
    {"url": "https://www.ticketweb.com/events", "type": "event"},
    {"url": "https://www.eventful.com/", "type": "event"},
    {"url": "https://www.seetickets.com/", "type": "event"},
    {"url": "https://www.ticketmaster.com/discover/events", "type": "event"},
    {"url": "https://www.eventbrite.com/d/online/events/", "type": "event"},
    {"url": "https://www.livenation.com/events", "type": "event"},
    {"url": "https://www.eventfinda.com.au/", "type": "event"},
    {"url": "https://www.ticketleap.com/events/", "type": "event"},
    {"url": "https://www.eventticketscenter.com/", "type": "event"},
    {"url": "https://www.eventzilla.net/", "type": "event"},
    {"url": "https://www.bandsintown.com/", "type": "event"},
    {"url": "https://www.songkick.com/", "type": "event"},
    {"url": "https://www.stubhub.com/concert-tickets", "type": "event"},
    {"url": "https://www.axs.com/events", "type": "event"},
    {"url": "https://www.dice.fm/events", "type": "event"},

    # Food (expanded)
    {"url": "https://www.delish.com/recipes/", "type": "food"},
    {"url": "https://www.thekitchn.com/recipes", "type": "food"},
    {"url": "https://www.foodandwine.com/recipes", "type": "food"},
    {"url": "https://www.cookinglight.com/recipes", "type": "food"},
    {"url": "https://www.taste.com.au/recipes", "type": "food"},
    {"url": "https://www.myrecipes.com/", "type": "food"},
    {"url": "https://www.kingarthurbaking.com/recipes", "type": "food"},
    {"url": "https://www.budgetbytes.com/recipes/", "type": "food"},
    {"url": "https://www.jamieoliver.com/recipes/", "type": "food"},
    {"url": "https://www.yummly.com/recipes", "type": "food"},
    {"url": "https://www.allrecipes.com/recipes/", "type": "food"},
    {"url": "https://www.foodnetwork.com/recipes", "type": "food"},
    {"url": "https://www.seriouseats.com/recipes", "type": "food"},
    {"url": "https://www.epicurious.com/recipes", "type": "food"},
    {"url": "https://www.gimmesomeoven.com/", "type": "food"},

    # Sports (expanded)
    {"url": "https://www.bleacherreport.com/nfl", "type": "sports"},
    {"url": "https://www.si.com/nba", "type": "sports"},
    {"url": "https://www.sbnation.com/soccer", "type": "sports"},
    {"url": "https://www.mlb.com/news", "type": "sports"},
    {"url": "https://www.nhl.com/news", "type": "sports"},
    {"url": "https://www.yardbarker.com/", "type": "sports"},
    {"url": "https://www.espncricinfo.com/", "type": "sports"},
    {"url": "https://www.cbssports.com/soccer/", "type": "sports"},
    {"url": "https://www.nbcsports.com/nfl", "type": "sports"},
    {"url": "https://www.athlonsports.com/", "type": "sports"},
    {"url": "https://www.nfl.com/", "type": "sports"},
    {"url": "https://www.nba.com/", "type": "sports"},
    {"url": "https://www.fifa.com/", "type": "sports"},
    {"url": "https://www.uefa.com/", "type": "sports"},
    {"url": "https://www.olympics.com/", "type": "sports"},

    # Portfolio (expanded)
    {"url": "https://www.coroflot.com/portfolios", "type": "portfolio"},
    {"url": "https://www.crevado.com/portfolios", "type": "portfolio"},
    {"url": "https://www.portfoliobox.net/", "type": "portfolio"},
    {"url": "https://www.fabrik.io/portfolios", "type": "portfolio"},
    {"url": "https://www.jouroportfolio.com/", "type": "portfolio"},
    {"url": "https://www.wix.com/portfolio-websites", "type": "portfolio"},
    {"url": "https://www.squarespace.com/portfolios", "type": "portfolio"},
    {"url": "https://www.pixpa.com/portfolios", "type": "portfolio"},
    {"url": "https://www.dunked.com/", "type": "portfolio"},
    {"url": "https://www.foliocollaborative.org/", "type": "portfolio"},
    {"url": "https://www.behance.net/discover", "type": "portfolio"},
    {"url": "https://www.dribbble.com/shots", "type": "portfolio"},
    {"url": "https://www.artstation.com/artwork", "type": "portfolio"},
    {"url": "https://www.carbonmade.com/", "type": "portfolio"},
    {"url": "https://www.format.com/portfolios", "type": "portfolio"},

    # Directory (expanded)
    {"url": "https://www.manta.com/", "type": "directory"},
    {"url": "https://www.superpages.com/", "type": "directory"},
    {"url": "https://www.dexknows.com/", "type": "directory"},
    {"url": "https://www.cylex.us.com/", "type": "directory"},
    {"url": "https://www.brownbook.net/", "type": "directory"},
    {"url": "https://www.whitepages.com/", "type": "directory"},
    {"url": "https://www.merchantcircle.com/", "type": "directory"},
    {"url": "https://www.showmelocal.com/", "type": "directory"},
    {"url": "https://www.hotfrog.com/", "type": "directory"},
    {"url": "https://www.bizhwy.com/", "type": "directory"},
    {"url": "https://www.yelp.com/search", "type": "directory"},
    {"url": "https://www.yellowpages.com/", "type": "directory"},
    {"url": "https://www.foursquare.com/explore", "type": "directory"},
    {"url": "https://www.local.com/", "type": "directory"},
    {"url": "https://www.mapquest.com/us/restaurants", "type": "directory"},

    # Real estate (expanded)
    {"url": "https://www.century21.com/real-estate/", "type": "real estate"},
    {"url": "https://www.remax.com/homes-for-sale", "type": "real estate"},
    {"url": "https://www.coldwellbankerhomes.com/", "type": "real estate"},
    {"url": "https://www.kw.com/homes-for-sale", "type": "real estate"},
    {"url": "https://www.sothebysrealty.com/eng/sales", "type": "real estate"},
    {"url": "https://www.bhhs.com/homes-for-sale", "type": "real estate"},
    {"url": "https://www.compass.com/listings/", "type": "real estate"},
    {"url": "https://www.elliman.com/properties", "type": "real estate"},
    {"url": "https://www.howardhanna.com/homes-for-sale", "type": "real estate"},
    {"url": "https://www.longandfoster.com/homes-for-sale", "type": "real estate"},
    {"url": "https://www.zillow.com/homes/", "type": "real estate"},
    {"url": "https://www.redfin.com/homes-for-sale", "type": "real estate"},
    {"url": "https://www.trulia.com/for_sale/", "type": "real estate"},
    {"url": "https://www.homes.com/for-sale/", "type": "real estate"},
    {"url": "https://www.realtor.com/realestateandhomes-for-sale/", "type": "real estate"},

    # Personal (expanded)
    {"url": "https://www.wordpress.com/create-blog/", "type": "personal"},
    {"url": "https://www.blogger.com/about/", "type": "personal"},
    {"url": "https://www.jimdo.com/website/personal/", "type": "personal"},
    {"url": "https://www.site123.com/personal-websites", "type": "personal"},
    {"url": "https://www.webnode.com/en/create/personal-website", "type": "personal"},
    {"url": "https://www.webflow.com/made-in-webflow/personal", "type": "personal"},
    {"url": "https://www.zoho.com/sites/personal/", "type": "personal"},
    {"url": "https://www.moonfruit.com/personal-websites", "type": "personal"},
    {"url": "https://www.doodlekit.com/personal-websites", "type": "personal"},
    {"url": "https://www.websitebuilder.com/personal/", "type": "personal"},
    {"url": "https://www.about.me/", "type": "personal"},
    {"url": "https://www.strikingly.com/s/create-personal-website", "type": "personal"},
    {"url": "https://www.carrd.co/", "type": "personal"},
    {"url": "https://www.gravatar.com/", "type": "personal"},
    {"url": "https://www.pattern.com/", "type": "personal"}, # Etsy personal shop site

    # NEW CATEGORY: AI/Tech Service (For company sites focused on AI products/services)
    {"url": "https://www.openai.com/", "type": "ai/tech service"},
    {"url": "https://www.ibm.com/cloud/ai", "type": "ai/tech service"},
    {"url": "https://www.google.ai/", "type": "ai/tech service"},
    {"url": "https://www.nvidia.com/en-us/deep-learning-ai/", "type": "ai/tech service"},
    {"url": "https://www.salesforce.com/products/ai/overview/", "type": "ai/tech service"},
    {"url": "https://www.microsoft.com/en-us/ai", "type": "ai/tech service"},
    {"url": "https://www.adobe.com/sensei/overview.html", "type": "ai/tech service"}, # Adobe Sensei
    {"url": "https://www.huggingface.co/", "type": "ai/tech service"},
    {"url": "https://www.aws.amazon.com/machine-learning/", "type": "ai/tech service"},
    {"url": "https://www.deepmind.com/", "type": "ai/tech service"},
    {"url": "https://www.datarobot.com/", "type": "ai/tech service"},
    {"url": "https://www.tableau.com/solutions/ai-machine-learning", "type": "ai/tech service"},
    {"url": "https://www.palantir.com/", "type": "ai/tech service"},
    {"url": "https://www.c3.ai/", "type": "ai/tech service"},
    {"url": "https://www.ai.google/", "type": "ai/tech service"},

    # NEW CATEGORY: Finance/Banking
    {"url": "https://www.bankofamerica.com/", "type": "finance/banking"},
    {"url": "https://www.chase.com/", "type": "finance/banking"},
    {"url": "https://www.wellsfargo.com/", "type": "finance/banking"},
    {"url": "https://www.citibank.com/", "type": "finance/banking"},
    {"url": "https://www.fidelity.com/trading/online-trading", "type": "finance/banking"},
    {"url": "https://www.schwab.com/investing", "type": "finance/banking"},
    {"url": "https://www.investopedia.com/trading", "type": "finance/banking"},
    {"url": "https://www.nerdwallet.com/banking/", "type": "finance/banking"},
    {"url": "https://www.money.com/investing/", "type": "finance/banking"},
    {"url": "https://www.fool.com/investing-news/", "type": "finance/banking"},
    {"url": "https://www.capitalone.com/", "type": "finance/banking"},
    {"url": "https://www.usbank.com/", "type": "finance/banking"},
    {"url": "https://www.morganstanley.com/", "type": "finance/banking"},
    {"url": "https://www.goldmansachs.com/", "type": "finance/banking"},
    {"url": "https://www.robinhood.com/", "type": "finance/banking"},

    # NEW CATEGORY: Reference/Database (for general information, dictionaries, encyclopedias)
    {"url": "https://www.britannica.com/", "type": "reference/database"},
    {"url": "https://www.merriam-webster.com/", "type": "reference/database"},
    {"url": "https://www.dictionary.com/", "type": "reference/database"},
    {"url": "https://www.imdb.com/", "type": "reference/database"},
    {"url": "https://www.rottentomatoes.com/", "type": "reference/database"},
    {"url": "https://www.collinsdictionary.com/", "type": "reference/database"},
    {"url": "https://www.oxfordreference.com/", "type": "reference/database"},
    {"url": "https://www.ncbi.nlm.nih.gov/pubmed/", "type": "reference/database"}, # Medical database
    {"url": "https://www.jstor.org/", "type": "reference/database"}, # Academic database
    {"url": "https://www.gutenberg.org/", "type": "reference/database"}, # Ebook database
    {"url": "https://www.archive.org/", "type": "reference/database"}, # Internet Archive
    {"url": "https://www.worldcat.org/", "type": "reference/database"}, # Library catalog
    {"url": "https://www.allmusic.com/", "type": "reference/database"}, # Music database
    {"url": "https://www.discogs.com/", "type": "reference/database"}, # Music database
    {"url": "https://www.goodreads.com/", "type": "reference/database"}, # Book database

    # NEW CATEGORY: Software/Product (for software products, downloads, or SaaS)
    {"url": "https://www.adobe.com/creativecloud.html", "type": "software/product"},
    {"url": "https://www.microsoft.com/en-us/software/windows", "type": "software/product"},
    {"url": "https://www.libreoffice.org/download/", "type": "software/product"},
    {"url": "https://www.gimp.org/downloads/", "type": "software/product"},
    {"url": "https://www.mozilla.org/en-US/firefox/new/", "type": "software/product"},
    {"url": "https://www.slack.com/downloads", "type": "software/product"},
    {"url": "https://www.zoom.us/download", "type": "software/product"},
    {"url": "https://www.spotify.com/us/download/windows/", "type": "software/product"},
    {"url": "https://www.audacityteam.org/download/", "type": "software/product"},
    {"url": "https://www.notepad-plus-plus.org/downloads/", "type": "software/product"},
    {"url": "https://www.figma.com/downloads/", "type": "software/product"},
    {"url": "https://www.autodesk.com/products/autocad/overview", "type": "software/product"},
    {"url": "https://www.blender.org/download/", "type": "software/product"},
    {"url": "https://www.jetbrains.com/pycharm/download/", "type": "software/product"},
    {"url": "https://code.visualstudio.com/download", "type": "software/product"}, # VS Code itself
]

df = pd.DataFrame(training_data_raw)

# --- 2. Scrape Content ---
scraped_texts = []
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

print("Scraping content for training data (this may take a while and show errors for some URLs)...")
for index, row in df.iterrows():
    url = row['url']
    try:
        # Ensure URL has scheme for requests
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract text, title, and meta description
        text = soup.get_text(separator=" ", strip=True).lower()
        title = soup.find("title").get_text().lower() if soup.find("title") else ""
        meta_description = ""
        for tag in soup.find_all("meta"):
            if tag.get("name") == "description":
                meta_description = tag.get("content", "").lower()
                break
        
        # Clean up excessive whitespace
        combined_text = f"{title} {meta_description} {text}"
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        # Limit text length to prevent memory issues with very large pages
        scraped_texts.append(combined_text[:10000]) # Take first 10,000 characters
        print(f"Scraped successfully: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}. Skipping this URL for training.")
        scraped_texts.append("") # Append empty string if scraping fails
    except Exception as e:
        print(f"Unexpected error processing {url}: {e}. Skipping this URL for training.")
        scraped_texts.append("")
    
    # Be polite: add a random delay between requests
    time.sleep(random.uniform(0.5, 2.0))

df['combined_text'] = scraped_texts

# Filter out rows where scraping failed completely or resulted in very little text
# This step is crucial to avoid training on empty data or causing stratify errors
initial_rows = len(df)
df = df[df['combined_text'].apply(lambda x: len(x) > 100)] # Require at least 100 characters of text
if len(df) < initial_rows:
    print(f"Warning: Filtered out {initial_rows - len(df)} URLs due to scraping errors or insufficient content.")

# Verify that all classes still have at least 2 samples after filtering
class_counts = df['type'].value_counts()
problematic_classes = class_counts[class_counts < 2].index.tolist()

if not df.empty and not problematic_classes:
    print(f"Data prepared successfully. {len(df)} samples remaining for training.")
    # --- 3. Feature Engineering ---
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['combined_text'])
    y = df['type']

    # --- 4. Train Model ---
    print("Training Logistic Regression model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    # --- 5. Evaluate Model ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete. Test Accuracy: {accuracy:.4f}")

    # --- 6. Save Model and Vectorizer ---
    print("Saving model and vectorizer...")
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(model, 'website_classifier_model.pkl')
    print("Model and vectorizer saved as tfidf_vectorizer.pkl and website_classifier_model.pkl")
    print("--- Training Script Finished ---")
else:
    if df.empty:
        print("Error: No meaningful data left after scraping and filtering. Cannot train model.")
    if problematic_classes:
        print(f"Error: The following classes have fewer than 2 samples after scraping/filtering: {problematic_classes}. Cannot perform stratified split.")
        print("Please ensure each website type has at least two robust URLs in your training_data_raw.")
    print("--- Training Script Failed ---")
