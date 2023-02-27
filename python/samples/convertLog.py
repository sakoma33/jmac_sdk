import json
class ConvertLog:
	BASE_URL = "https://tenhou.net/5/#json="
	def __init__(self):
		self.logs = { "title":["",""], "name":["","","",""], "rule":{"disp":"","aka":1}, "log":[]}

	def add_log(self,obs_dict):
		convert_id = lambda id: 50+id//36+1 if id in [16, 52, 88] else (id//36+1)*10+(id%36)//4+1
		log = [[0,0,0],[0,0,0,0],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		for i,obs in enumerate(obs_dict.values()):
			obs=json.loads(obs.to_json())
			if i == 0:
				self.logs["name"] = obs["publicObservation"]["playerIds"]
				
				if "round" in obs["publicObservation"]["initScore"].keys():
					log[0][0] = obs["publicObservation"]["initScore"]["round"]

				if "honba" in obs["publicObservation"]["initScore"].keys():
					log[0][1] = obs["publicObservation"]["initScore"]["honba"]

				if "riichi" in obs["publicObservation"]["initScore"].keys():
					log[0][2] = obs["publicObservation"]["initScore"]["riichi"]

				log[1] = obs["publicObservation"]["initScore"]["tens"]

				for dora in obs["publicObservation"]["doraIndicators"]:
					log[2].append(convert_id(dora))

				if "noWinner" in obs["roundTerminal"].keys():
					log.append(["",obs["roundTerminal"]["noWinner"]["tenChanges"]])
				else:
					#todo:包の処理
					for win in obs["roundTerminal"]["wins"]:
						if "uraDoraIndicators" in win.keys():
							log[3] = win["uraDoraIndicators"]
						log.append(["",win["tenChanges"]])
						who_win = win["who"] if "who" in win.keys() else 0
						from_who_win = win["fromWho"] if "fromWho" in win.keys() else 0
						log.append([who_win,from_who_win,who_win,str(win["ten"])])
			who=obs["who"] if "who" in obs.keys() else 0
			log[4+who*3] = [convert_id(id) for id in obs["privateObservation"]["initHand"]["closedTiles"]]
			tumo_count=0
			event_count=0
			while event_count<len(obs["publicObservation"]["events"]):
				event = obs["publicObservation"]["events"][event_count]
				who_event = event["who"] if "who" in event.keys() else 0
				if who == who_event:
					if not "type" in event.keys():
						tile = event["tile"] if "tile" in event.keys() else 0
						log[4+who*3+2].append(convert_id(tile))

					elif event["type"] == "EVENT_TYPE_TSUMOGIRI":
						log[4+who*3+2].append(60)

					elif event["type"] == "EVENT_TYPE_DRAW":
						log[4+who*3+1].append(convert_id(obs["privateObservation"]["drawHistory"][tumo_count]))
						tumo_count += 1

					elif event["type"] == "EVENT_TYPE_RIICHI":
						event_count+=1
						next_event=obs["publicObservation"]["events"][event_count]
						tile = next_event["tile"] if "tile" in next_event.keys() else 0
						log[4+who*3+2].append('r'+str(convert_id(tile)))


					elif event["type"] == "EVENT_TYPE_CHI":
						open = event["open"]
						chi_offset = [((0b0000000000011000 & open)>>3)%4,
		       						  ((0b0000000001100000 & open)>>5)%4,
									  ((0b0000000110000000 & open)>>7)%4,
		       						 ]
						chi_base_and_stolen = (0b1111110000000000 & open)>>10
						stolen = chi_base_and_stolen%3
						chi_base = ((chi_base_and_stolen//3)%7)*4+((chi_base_and_stolen//3)//7)*36
						open_tile = [chi_base+stolen*4+chi_offset[stolen],
		   							 min(chi_base+((stolen+1)%3)*4+chi_offset[((stolen+1)%3)],chi_base+((stolen+2)%3)*4+chi_offset[((stolen+2)%3)]),
									 max(chi_base+((stolen+1)%3)*4+chi_offset[((stolen+1)%3)],chi_base+((stolen+2)%3)*4+chi_offset[((stolen+2)%3)])
									]
						log[4+who*3+1].append('c'+ str(convert_id(open_tile[0])) + str(convert_id(open_tile[1])) + str(convert_id(open_tile[2])))

					elif event["type"] == "EVENT_TYPE_PON":
						open = event["open"]
						mask_from = open%4
						pon_unused_offset = ((0b0000000001100000 & open)>>5)%3
						pon_base_and_stolen = (0b1111111000000000 & open)>>9
						pon_base = (pon_base_and_stolen//3)*4
						stolen = pon_base_and_stolen%3
						open_tile = list(range(pon_base,pon_base+4))
						open_tile.pop(pon_unused_offset)
						stolen_tile = open_tile.pop(stolen)
						open_tile = [str(convert_id(id)) for id in open_tile]
						open_tile.insert(3-mask_from,'p'+str(convert_id(stolen_tile)))
						log[4+who*3+1].append(''.join(open_tile))

					elif event["type"]=="EVENT_TYPE_CLOSED_KAN":
						open = event["open"] if "open" in event.keys() else 0
						kan_tile = open>>8
						# aを置く位置よく分からん（違うとバグる）
						# 赤の位置も確認が必要
						open_tile=[str(convert_id((kan_tile//4)*4+1)),
		 						   str(convert_id((kan_tile//4)*4)),
								   str(convert_id((kan_tile//4)*4+2)),
								   'a',
								   str(convert_id((kan_tile//4)*4+3))
								  ]
						log[4+who*3+2].append(''.join(open_tile))
					
					elif event["type"]=="EVENT_TYPE_ADDED_KAN":
						open = event["open"]
						mask_from = open%4
						pon_unused_offset = ((0b0000000001100000 & open)>>5)%3
						pon_base_and_stolen = (0b1111111000000000 & open)>>9
						pon_base = (pon_base_and_stolen//3)*4
						stolen = pon_base_and_stolen%3
						open_tile = list(range(pon_base,pon_base+4))
						added_tile=open_tile.pop(pon_unused_offset)
						stolen_tile = open_tile.pop(stolen)
						open_tile.append(added_tile)
						open_tile = [str(convert_id(id)) for id in open_tile]
						open_tile.insert(3-mask_from,'k'+str(convert_id(stolen_tile)))
						log[4+who*3+2].append(''.join(open_tile))

					elif event["type"]=="EVENT_TYPE_OPEN_KAN":
						open = event["open"]
						mask_from = open%4
						kan_tile = open>>8
						open_tile = list(range((kan_tile//4)*4,(kan_tile//4)*4+4))
						stolen_tile = open_tile.pop(kan_tile%4)
						open_tile = [str(convert_id(id)) for id in open_tile]
						insert_pos = 3-mask_from if mask_from>=2 else 3
						open_tile.insert(insert_pos,'m'+str(convert_id(stolen_tile)))
						log[4+who*3+1].append(''.join(open_tile))
						log[4+who*3+2].append(0)
				
				event_count+=1						

		self.logs["log"].append(log)
	
	def get_url(self):
		return self.BASE_URL+json.dumps(self.logs, separators=(',', ':'))