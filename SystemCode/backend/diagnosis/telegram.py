import telegram

TOKEN = '6718101329:AAE-AI14PUTSVbMUnZaDXL5D1Ff-qg9lDM0'

async def filter_msg(msg):
    bot = telegram.Bot(TOKEN)
    
    found, chat_id = False, None
    async with bot:
        updates = await bot.get_updates()

        for update in updates:
            if msg == update.message.text:
                found = True
                chat_id = update.message.from_user.id
                break
    return found, chat_id


async def send_msg(chat_id, msg):
    bot = telegram.Bot(TOKEN)
    
    async with bot:
        await bot.send_message(chat_id=chat_id, text=msg)