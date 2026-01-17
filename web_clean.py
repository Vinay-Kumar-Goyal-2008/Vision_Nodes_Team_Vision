
def search_youtube(search_query):
    
    page.goto(f"https://www.youtube.com/results?search_query={search_query}&sp=EgIIAQ%253D%253D")
    
def play_video():
    page.locator("ytd-video-renderer").first.click()

def skip_ad():
    try:
    
        btn = page.locator(".ytp-ad-skip-button, .ytp-ad-overlay-close-button")
        btn.wait_for(state="visible", timeout=5000)
        btn.click()
    except:
        pass

def play_pause():
    """Play/Pause: k or Spacebar"""
    focus_player()
    page.keyboard.press('k')

def voice_search():
    """Click the YouTube voice search button"""
    page.locator("button[aria-label='Search with your voice']").click()

def volume_up_pyautogui():
    """Increase system volume using PyAutoGUI"""
    import pyautogui
    pyautogui.press('volumeup')

def volume_down_pyautogui():
    """Decrease system volume using PyAutoGUI"""
    import pyautogui
    pyautogui.press('volumedown')

def volume_mute_pyautogui():
    """Mute/Unmute system volume using PyAutoGUI"""
    import pyautogui
    pyautogui.press('volumemute')


# GMAIL

def filter_unread():
    page.goto("https://mail.google.com/mail/u/0/#search/label%3Aunread")

def read_email(): # think
    count = page.locator("tr.zA").count()
    for i in range(min(3, count)):
        row = page.locator("tr.zA").nth(i) 
        row.click()

    # div.a3s = Message Body
    content = page.locator("div.a3s").first.inner_text()
    print(content) # read aloud here
    page.go_back()

def compose_email():
    """Compose: c"""
    page.keyboard.press('c') # pyautogui


def archive_email():
    """Archive: e"""
    page.keyboard.press('e')

def jump_to_inbox():
    """Jump to Inbox: g then i"""
    page.keyboard.press('g')
    page.keyboard.press('i')

def read_aloud_playwright():
    """Toggle Edge Read Aloud: Ctrl+Shift+U (Playwright)"""
    page.keyboard.press('Control+Shift+u')

def read_aloud_pyautogui():
    """Toggle Edge Read Aloud: Ctrl+Shift+U (PyAutoGUI)"""
    import pyautogui
    pyautogui.hotkey('ctrl', 'shift', 'u')

def window_voice_typing():
    """Toggle Windows Voice Typing: Win+H (PyAutoGUI)"""
    import pyautogui
    pyautogui.hotkey('win', 'h')

def spam_mail():
    """Navigate to Gmail Spam folder"""
    page.goto("https://mail.google.com/mail/u/0/#spam")

def get_spam_count():
    """Get the unread spam count from Gmail sidebar"""
    count_element = page.locator("a[aria-label^='Spam'] + div.bsU")
    if count_element.is_visible():
        return int(count_element.inner_text())
    return 0



2TYJCsiaY3x86YP9Tb7ZbGg25u8mPfB2