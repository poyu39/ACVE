

class Ass:
    def __init__(self, ass_path) -> None:
        '''
            Ass subtitles file
        '''
        self.raw = []
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            # self.raw = f.readlines()
            for line in f.readlines():
                self.raw.append(line.replace('\n', ''))
    
    def format_events(self) -> None:
        '''
            [Events]
            Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        '''
        event_start_flag = False
        self.dialogues = []
        for line in self.raw:
            if line != '[Events]' and event_start_flag == False:
                continue
            elif line == '[Events]':
                event_start_flag = True
                continue
            if event_start_flag:
                if line == '; --- ;':
                    continue
                try:
                    format_layer, start, end, style, name, margin_l, margin_r, margin_v, effect, text = line.split(',')
                    format, layer = format_layer.split(': ')
                    if format == 'Dialogue':
                        self.dialogues.append(
                            {
                                'start': start,
                                'end': end,
                                'style': style,
                                'text': text,
                            }
                        )
                except:
                    continue
    
    def get_dialogues(self, style) -> list:
        return [dialogue for dialogue in self.dialogues if dialogue['style'] == style]