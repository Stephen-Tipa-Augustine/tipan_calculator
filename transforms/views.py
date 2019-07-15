from django.shortcuts import render, redirect
from django.http import HttpResponse
from .functions import functions 

# Create your views here.

def receive(request):

	return HttpResponse("We have you baby")
	
def transforms(request, slug):
	function = slug
	answer = functions.transforms(f=slug, kind=9)
	return HttpResponse(answer)
	
def start_page(request):

	return HttpResponse("Welcome to our backend site baby")