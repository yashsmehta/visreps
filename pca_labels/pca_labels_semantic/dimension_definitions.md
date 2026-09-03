# Six binary dimensions for ImageNet-1k classes

Answer each question for the TYPICAL instance of the class as it appears in a typical photo.
Every value is 0 or 1. Use plain common sense. Do not use WordNet or any taxonomy tool.

## natural (1 = natural, 0 = manmade)
Did it grow or occur in nature (1), or was it built, manufactured, or cooked by people (0)?
1: all animals, people (ballplayer, groom, scuba diver), plants, flowers, fungi, raw fruit and vegetables,
   corn, acorn, hay, landscapes (alp, cliff, valley, seashore, volcano, coral reef, geyser, lakeside...),
   honeycomb, spider web, bubble.
0: every artifact, building, vehicle, tool, garment, AND prepared or cooked food (pizza, cheeseburger,
   hotdog, burrito, ice cream, bagel, French loaf, pretzel, meat loaf, trifle, consomme, guacamole,
   mashed potato, carbonara, chocolate sauce, dough, eggnog, espresso, red wine).

## handheld (1 = handheld, 0 = larger)
Could an ordinary adult pick it up and hold it in their hands or arms (roughly under 10 kg and
smaller than a large suitcase)?
1: Chihuahua, cat, chicken, rabbit, hamster, most birds, insects, snakes, small fish, frogs, cups,
   cameras, phones, guitars, hammers, shoes, hats, balls, fruit, bread, laptops, small dogs.
0: Labrador, German shepherd, husky and other dogs heavier than ~12 kg, goat, horse, big cats, bears,
   sharks, ostrich, crocodiles, sofas, fridges, tables, beds, vehicles, buildings, landscapes,
   grand piano, cello, pool table, mountain tent, canoe, bicycles.
Dog breeds: judge by typical adult breed weight. Toy and small breeds (terriers, spaniels, pugs,
Pomeranian, Maltese, Pekingese, Shih-Tzu, Papillon, Chihuahua, dachshund, beagle, corgi...) = 1.
Medium and large breeds (retrievers, shepherds, hounds over 12 kg, mastiffs, huskies, collies,
setters, pointers, boxers, Dobermans, Great Danes, wolves, foxes, hyenas...) = 0.

## indoor (1 = indoor, 0 = outdoor)
Where would you normally encounter it: inside a building or home (1), or outside (0)?
1: all dog breeds and house pets (cats, hamster, guinea pig, goldfish, rabbit), furniture,
   appliances, kitchenware, prepared food and drinks, fruit and vegetables (as bought and eaten),
   clothing, tools, musical instruments, electronics, books, toys, indoor sports gear
   (basketball, ping-pong ball, pool table, dumbbell, barbell, punching bag).
0: wild animals, farm animals, wolves and foxes, plants and flowers growing, fungi, landscapes,
   vehicles, buildings, bridges, fences, monuments, outdoor sports gear (soccer ball, rugby ball,
   golf ball, ski, bobsled, racket), garden tools (shovel, lawn mower, plow), weapons like cannon
   and rifle, outdoor structures (pier, dock, fountain, tent, yurt).

## self_moving (1 = moves on its own, 0 = stationary)
Does it move around by its own power, meaning muscles, an engine, a motor, sails, or lift?
1: all animals and people; all vehicles and craft (cars, trucks, trains, boats, ships, planes,
   airship, balloon, submarine, missile, space shuttle, tractors, harvesters, forklifts, snowplows,
   golfcart, snowmobile, mopeds, motor scooters) including human-powered vehicles (bicycles,
   unicycle, canoe, gondola, jinrikisha, horse cart, oxcart, dogsled, bobsled, shopping cart, barrow).
0: plants, fungi, food, landscapes, buildings, furniture, tools, instruments, clothing, electronics,
   and everything else that stays where it is put. Car parts (car wheel, car mirror, grille,
   oil filter, disk brake) = 0.

## soft (1 = soft, 0 = rigid)
Would it squash, bend, or deform if you pressed on it firmly with your hand?
1: furry or feathered animals, fish, frogs, snakes, worms, slugs, jellyfish, people, flowers,
   leaves, soft fruit (banana, strawberry, fig, orange), bread and prepared food, textiles,
   clothing, shoes made of fabric, bags, pillows, quilts, curtains, rugs, towels, wigs, sponges,
   paper items, plastic bags, balloons, rubber balls, mattresses, sofas.
0: insects, beetles, crabs, lobsters, turtles, tortoises, snails (shell), armadillo, everything
   made of metal, wood, glass, stone, ceramic, or hard plastic, hard fruit and vegetables
   (pineapple, acorn squash, coconut-like things), landscapes, buildings, vehicles, furniture
   made of wood or metal, instruments, tools, electronics.

## elongated (1 = elongated, 0 = compact)
Is its typical shape at least about 2.5 times longer in one direction than in the others?
1: snakes, eels, worms, centipedes, dachshund, herons, cranes, flamingos, storks, giraffe-like tall
   animals, alligators, lizards, trains, buses, limousines, ships, submarines, missiles, airliners,
   bridges, viaducts, obelisks, flagpoles, totem poles, towers, pens, pencils, screwdrivers,
   hammers, knives, cleavers, rifles, swords, bows, flutes, oboes, trombones, cellos, guitars,
   baseball bats, paddles, oars, skis, ties, socks, scarves, feather boas, belts, chains, cucumbers,
   zucchinis, bananas, corn ears, French loaf, hotdog, cigarettes, matchsticks, candles,
   toothbrushes, syringes, crutches, ladders, park benches, canoes, kayaks.
0: cats, most dogs, bears, birds with compact bodies, fish with compact bodies, frogs, turtles,
   spiders, balls, cups, bowls, cameras, phones, laptops, clocks, chairs, sofas, tables, cars,
   houses, apples, oranges, pizzas, hats, shoes, bags, boxes, drums, pianos, landscapes.
