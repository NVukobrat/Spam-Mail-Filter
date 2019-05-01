import os
import random
import nltk


class EmailType:
    SPAM = 1
    HAM = 0


class SpamClf:
    pA = 0
    pNotA = 0
    positiveTotal = 0
    negativeTotal = 0
    trainPositive = {}
    trainNegative = {}

    def readDataset(self, hamPath, spamPath):
        return self.readEmails(hamPath, EmailType.HAM) + self.readEmails(spamPath, EmailType.SPAM)

    def readEmails(self, path, type):
        emails = []
        for emailFile in os.listdir(path):
            with open(path + emailFile, "r", errors="replace") as files:
                for email in files:
                    emails.append([email, type])

        return emails

    def splitDataset(self, dataset, ratio):
        splitBoundary = int(len(dataset) * ratio)
        random.Random(1).shuffle(dataset)

        return dataset[:splitBoundary], dataset[splitBoundary:]

    def train(self, dataset):
        total = 0
        numSpam = 0
        for email in dataset:
            if email[1] == EmailType.SPAM:
                numSpam += 1
            total += 1
            self.processEmail(email[0], email[1])
        self.pA = numSpam / float(total)
        self.pNotA = (total - numSpam) / float(total)
        # self.dropIrrelevantWords() # Slowing down algorithm, higher error rate

    def processEmail(self, body, label):
        processedEmail = self.structureEmail(body)

        for word in processedEmail:
            if label == EmailType.SPAM:
                self.trainPositive[word] = self.trainPositive.get(word, 0) + 1
                self.positiveTotal += 1
            else:
                self.trainNegative[word] = self.trainNegative.get(word, 0) + 1
                self.negativeTotal += 1

    def structureEmail(self, body):
        processedEmail = []
        for word in body.split():
            if len(word) < 3:
                continue
            word = ''.join(c for c in word if c.isalnum())  # No difference in results
            word = nltk.WordNetLemmatizer().lemmatize(word)  # No difference in results

            processedEmail.append(word)

        processedEmail += self.findBigramsByWord(processedEmail)

        return processedEmail

    def findBigramsByWord(self, body):
        bigramList = []
        for i in range(len(body) - 2):
            bigramList.append((body[i] + " " + body[i + 1]))
        return bigramList

    def findBigramsByChars(self, body):
        bigramList = []
        fullSentence = " ".join(str(e) for e in body)
        fullStrippedSentence = fullSentence.strip().replace(" ", "")

        for i in range(len(fullStrippedSentence) - 2):
            bigramList.append(fullStrippedSentence[i] + fullStrippedSentence[i + 1])

        return bigramList

    def findTrigramsByWord(self, body):
        bigramList = []
        for i in range(len(body) - 3):
            bigramList.append((body[i] + " " + body[i + 1]) + " " + body[i + 2])
        return bigramList

    def findTrigramsByChars(self, body):
        trigramList = []
        fullSentence = " ".join(str(e) for e in body)
        fullStrippedSentence = fullSentence.strip().replace(" ", "")

        for i in range(len(fullStrippedSentence) - 3):
            trigramList.append(fullStrippedSentence[i] + fullStrippedSentence[i + 1] + fullStrippedSentence[i + 2])

        return trigramList

    def dropIrrelevantWords(self):
        self.dropIrrelevantWordsFromCorpus(self.trainPositive)
        self.dropIrrelevantWordsFromCorpus(self.trainNegative)

    def dropIrrelevantWordsFromCorpus(self, corpus):
        for word in corpus:
            wordApperienceNumber = corpus[word]
            totalWordNumber = len(corpus)
            occurrenceInPercentage = (round(((wordApperienceNumber / float(totalWordNumber)) * 100), 3))

            if occurrenceInPercentage <= 5:
                corpus[word] = 0
            elif 40 <= occurrenceInPercentage <= 60:
                corpus[word] = 0

        for key, value in list(corpus.items()):
            if value == 0:
                del corpus[key]


    def classify(self, email):
        isSpam = self.pA * self.conditionalEmail(email, True)
        isHam = self.pNotA * self.conditionalEmail(email, False)

        return EmailType.SPAM if (isSpam > isHam) else EmailType.HAM

    def conditionalEmail(self, body, spam):
        result = 1.0
        processedEmail = self.structureEmail(body)

        for word in processedEmail:
            result *= self.conditionalWord(word, spam)

        return result

    def conditionalWord(self, word, spam):
        if spam:
            return (self.trainPositive.get(word, 0) + 1) / float(self.positiveTotal + (self.positiveTotal + self.negativeTotal))
        return (self.trainNegative.get(word, 0) + 1) / float(self.negativeTotal + (self.positiveTotal + self.negativeTotal))


# TESTING #

clf = SpamClf()
dataset = clf.readDataset("data/ham/", "data/spam/")
trainData, testData = clf.splitDataset(dataset, 0.8)
clf.train(trainData)

# TEST ON TRAIN #
correct = 0
total = len(trainData)
label = [EmailType.SPAM, EmailType.HAM]
for email in trainData:
    prediction = clf.classify(email[0])
    if prediction == email[1]:
        correct += 1

print("Precision TST:", round(correct/total, 3) * 100, "%")

# TEST ON RANDOM #
correct = 0
total = len(testData)
for email in testData:
    prediction = random.choice(label)
    if prediction == email[1]:
        correct += 1

print("Precision RND:", round(correct/total, 3) * 100, "%")

# TEST ON TEST #
correct = 0
total = len(testData)
for email in testData:
    prediction = clf.classify(email[0])
    if prediction == email[1]:
        correct += 1

print("Precision CLF:", round(correct/total, 3) * 100, "%")
