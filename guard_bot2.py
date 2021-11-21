from rgkit import rg

# will move off spawn and guard
class Robot:
    def act(self, game):
        if 'spawn' in rg.loc_types(self.location):
            return ['move', rg.toward(self.location, rg.CENTER_POINT)]
        return ['guard']
