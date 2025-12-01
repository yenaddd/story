from django.contrib import admin

from .models import StoryNode, NodeChoice

# 관리자 페이지에 노드와 선택지를 등록합니다.
admin.site.register(StoryNode)
admin.site.register(NodeChoice)