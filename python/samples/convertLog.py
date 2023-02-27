class ConvertLog:
	BASE_URL = "https://tenhou.net/5/#json="
	def __init__(self):
		self.logs = { "title":["",""], "name":["","","",""], "rule":{"disp":"","aka":1}, "log":[]}

	def convert_id(self,id):
		if id in [16, 52, 88]:
			return 50+id//36+1
		else:
			return (id//36+1)*10+(id%36)//4+1

	def add_log(self,obs_dict):
		log = [[0,0,0],[0,0,0,0],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		for i,obs in enumerate(obs_dict.values()):
			if i == 0:
				if "round" in obs["publicObservation"]["initScore"].keys():
					log[0][0] = obs["publicObservation"]["initScore"]["round"]
				if "honba" in obs["publicObservation"]["initScore"].keys():
					log[0][1] = obs["publicObservation"]["initScore"]["honba"]
				if "riichi" in obs["publicObservation"]["initScore"].keys():
					log[0][2] = obs["publicObservation"]["initScore"]["riichi"]
				log[1] = obs["publicObservation"]["playerIds"]
				for dora in obs["publicObservation"]["doraIndicators"]:
					log[2].append(self.convert_id(dora))
				if "noWinner" in obs["roundTerminal"].keys():
					log.append(["流局",obs["roundTerminal"]["noWinner"]["tenChanges"]])
				else:
					#todo:包の処理
					for win in obs["roundTerminal"]["wins"]:
						if "uraDoraIndicators" in win.keys():
							log[3] = win["uraDoraIndicators"]
						log.append(["和了",win["tenChanges"]])
						log.append([win["who"],win["fromWho"],win["who"],str(win["ten"])+"点"])
			who=obs["who"]
			log[4+who*3] = [self.convert_id(id) for id in obs["privateObservation"]["initHand"]["closedTiles"]]
			count=0
			for event in obs["publicObservation"]["events"]:
				who_event=event["who"] if "who" in event.keys() else 0
				if who==who_event:
					if not "type" in event.keys():
						log[4+who*3+2].append(self.convert_id(event["tile"]))
					elif event["type"] == "EVENT_TYPE_TSUMOGIRI":
						log[4+who*3+2].append(60)
					elif event["type"] == "EVENT_TYPE_DRAW":
						log[4+who*3+1].append(obs["privateObservation"]["events"][count])
						count += 1
					elif event["type"] == "EVENT_TYPE_CHI":
						open=event["open"]
						chi_offset = [(0b0000000000011000 & open)>>3,
		       						  (0b0000000001100000 & open)>>5,
									  (0b0000000110000000 & open)>>7,
		       						 ]
						chi_base_and_stolen = (0b1111110000000000 & open)>>10
						stolen = chi_base_and_stolen%3
						chi_base = ((chi_base_and_stolen//3)%9)*4+((chi_base_and_stolen//3)//7)*72
						open_tile = [chi_base+stolen*4+chi_offset[stolen],
		   							 min(chi_base+((stolen+1)%3)*4+chi_offset[((stolen+1)%3)],chi_base+((stolen+2)%3)*4+chi_offset[((stolen+2)%3)]),
									 max(chi_base+((stolen+1)%3)*4+chi_offset[((stolen+1)%3)],chi_base+((stolen+2)%3)*4+chi_offset[((stolen+2)%3)])
									]
						log[4+who*3+1].append('c'+ str(self.convert_id(open_tile[0])) + str(self.convert_id(open_tile[1])) + str(self.convert_id(open_tile[2])))
						

			# log[4+who*3+1]
			# log[4+who*3+2]
			
		self.logs["log"].append(log)
	
	def get_url(self):
		return 1