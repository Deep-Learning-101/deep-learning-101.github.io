---
layout: default
title: "AI 事件快訊：大事發生後 48 小時內的短稿與後續更新"
title_seo: "AI 事件快訊 | Deep Learning 101"
description: "Deep Learning 101 的事件短稿欄目：AI 領域重大事件發生後 24 至 48 小時內先出短稿，講清楚發生什麼事、為什麼重要，同一篇持續更新，並連回本站的深度整理頁。"
permalink: /News
lang: zh-Hant
---

{% include header.html %}

---

{% include ai-share.html %}

---

# AI 事件快訊

這個欄目只做一件事：AI 領域有夠大的事件發生時，在 24 至 48 小時內先出一篇短稿，講清楚發生什麼事、為什麼重要、接下來看什麼。事件有後續發展就更新同一篇，不另開新文；想看完整脈絡的人，文末會連回本站的深度整理頁。

這裡不做每日新聞彙整，也不追逐每一則小道消息。一個月約二到四篇，只收旗艦模型發布、大型資安事件、影響台灣開發者與企業的重大變化。

## 最新短稿

{% assign news = site.pages | where_exp: "p", "p.path contains 'News/'" | sort: "date" | reverse %}
{% if news.size == 0 %}
目前尚無短稿，第一篇上線後會自動列在這裡。
{% else %}
{% for p in news %}
- **{{ p.date | date: "%Y-%m-%d" }}** [{{ p.title }}]({{ p.url }})：{{ p.description }}
{% endfor %}
{% endif %}
