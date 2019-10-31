#include <cmath>
#include "mtcnn.h"

bool cmpScore(FaceBox lsh, FaceBox rsh) {
	if (lsh.score < rsh.score)
		return true;
	else
		return false;
}

bool cmpArea(FaceBox lsh, FaceBox rsh) {
	if (lsh.area < rsh.area)
		return false;
	else
		return true;
}

MTCNN::MTCNN(const string &model_path="./models") {

	vector<string> param_files = {
		model_path+"/det1.param",
		model_path+"/det2.param",
		model_path+"/det3.param"
	};

	vector<string> bin_files = {
		model_path+"/det1.bin",
		model_path+"/det2.bin",
		model_path+"/det3.bin"
	};

	Pnet.load_param(param_files[0].data());
	Pnet.load_model(bin_files[0].data());
	Rnet.load_param(param_files[1].data());
	Rnet.load_model(bin_files[1].data());
	Onet.load_param(param_files[2].data());
	Onet.load_model(bin_files[2].data());
}

MTCNN::~MTCNN(){
    Pnet.clear();
    Rnet.clear();
    Onet.clear();
}

void MTCNN::SetMinFace(int minSize){
	minsize = minSize;
}

void MTCNN::generateFaceBox(ncnn::Mat score, ncnn::Mat location, vector<FaceBox>& boundingBox_, float scale){
    const int stride = 2;
    const int cellsize = 12;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
    //float *plocal = location.data;
    FaceBox FaceBox;
    float inv_scale = 1.0f/scale;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                FaceBox.score = *p;
                FaceBox.x1 = round((stride*col+1)*inv_scale);
                FaceBox.y1 = round((stride*row+1)*inv_scale);
                FaceBox.x2 = round((stride*col+1+cellsize)*inv_scale);
                FaceBox.y2 = round((stride*row+1+cellsize)*inv_scale);
                FaceBox.area = (FaceBox.x2 - FaceBox.x1) * (FaceBox.y2 - FaceBox.y1);
                const int index = row * score.w + col;
                for(int channel=0;channel<4;channel++){
                    FaceBox.regreCoord[channel]=location.channel(channel)[index];
                }
                boundingBox_.push_back(FaceBox);
            }
            p++;
            //plocal++;
        }
    }
}

void MTCNN::nmsTwoBoxs(vector<FaceBox>& boundingBox_, vector<FaceBox>& previousBox_, const float overlap_threshold, string modelname)
{
	if (boundingBox_.empty()) {
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	//cout << boundingBox_.size() << " ";
	for (vector<FaceBox>::iterator ity = previousBox_.begin(); ity != previousBox_.end(); ity++) {
		for (vector<FaceBox>::iterator itx = boundingBox_.begin(); itx != boundingBox_.end();) {
			int i = itx - boundingBox_.begin();
			int j = ity - previousBox_.begin();
			maxX = max(boundingBox_.at(i).x1, previousBox_.at(j).x1);
			maxY = max(boundingBox_.at(i).y1, previousBox_.at(j).y1);
			minX = min(boundingBox_.at(i).x2, previousBox_.at(j).x2);
			minY = min(boundingBox_.at(i).y2, previousBox_.at(j).y2);
			//maxX1 and maxY1 reuse
			maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two FaceBox
			IOU = maxX * maxY;
			if (!modelname.compare("Union"))
				IOU = IOU / (boundingBox_.at(i).area + previousBox_.at(j).area - IOU);
			else if (!modelname.compare("Min")) {
				IOU = IOU / ((boundingBox_.at(i).area < previousBox_.at(j).area) ? boundingBox_.at(i).area : previousBox_.at(j).area);
			}
			if (IOU > overlap_threshold&&boundingBox_.at(i).score>previousBox_.at(j).score) {
			//if (IOU > overlap_threshold) {
				itx = boundingBox_.erase(itx);
			}
			else {
				itx++;
			}
		}
	}
	//cout << boundingBox_.size() << endl;
}

void MTCNN::nms(vector<FaceBox> &boundingBox_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    vector<int> vPick;
    int nPick = 0;
    multimap<float, int> vScores;
    const int num_boxes = boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i){
		vScores.insert(pair<float, int>(boundingBox_[i].score, i));
	}
    while(vScores.size() > 0){
        int last = vScores.rbegin()->second;
        vPick[nPick] = last;
        nPick += 1;
        for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //maxX1 and maxY1 reuse 
            maxX = ((minX-maxX+1)>0)? (minX-maxX+1) : 0;
            maxY = ((minY-maxY+1)>0)? (minY-maxY+1) : 0;
            //IOU reuse for the area of two FaceBox
            IOU = maxX * maxY;
            if(!modelname.compare("Union"))
                IOU = IOU/(boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
            else if(!modelname.compare("Min")){
                IOU = IOU/((boundingBox_.at(it_idx).area < boundingBox_.at(last).area)? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
            }
            if(IOU > overlap_threshold){
                it = vScores.erase(it);
            }else{
                it++;
            }
        }
    }
    
    vPick.resize(nPick);
    vector<FaceBox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}

void MTCNN::refine(vector<FaceBox> &vecFaceBox, const int &height, const int &width, bool square){
    if(vecFaceBox.empty()){
        cout<<"FaceBox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<FaceBox>::iterator it=vecFaceBox.begin(); it!=vecFaceBox.end();it++){
        bbw = (*it).x2 - (*it).x1 + 1;
        bbh = (*it).y2 - (*it).y1 + 1;
        x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
        y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
        x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
        y2 = (*it).y2 + (*it).regreCoord[3]*bbh;

        
        
        if(square){
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);
        }

        //boundary check
        if((*it).x1<0)(*it).x1=0;
        if((*it).y1<0)(*it).y1=0;
        if((*it).x2>width)(*it).x2 = width - 1;
        if((*it).y2>height)(*it).y2 = height - 1;

        it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
    }
}

void MTCNN::extractMaxFace(vector<FaceBox>& boundingBox_)
{
	if (boundingBox_.empty()) {
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), cmpArea);
	for (vector<FaceBox>::iterator itx = boundingBox_.begin() + 1; itx != boundingBox_.end();) {
		itx = boundingBox_.erase(itx);
	}
}

void MTCNN::PNet(float scale)
{
	//first stage
	int hs = (int)ceil(img_h*scale);
	int ws = (int)ceil(img_w*scale);
	ncnn::Mat in;
	resize_bilinear(img, in, ws, hs);
	ncnn::Extractor ex = Pnet.create_extractor();
	ex.set_light_mode(true);
	//sex.set_num_threads(4);
	ex.input("data", in);
	ncnn::Mat score_, location_;
	ex.extract("prob1", score_);
	ex.extract("conv4-2", location_);
	vector<FaceBox> boundingBox_;

	generateFaceBox(score_, location_, boundingBox_, scale);
	nms(boundingBox_, nms_threshold[0]);

	firstFaceBox_.insert(firstFaceBox_.end(), boundingBox_.begin(), boundingBox_.end());
	boundingBox_.clear();
}

void MTCNN::PNet(){
    firstFaceBox_.clear();
    float minl = img_w < img_h? img_w: img_h;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = pre_facetor;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        scales_.push_back(m);
        minl *= factor;
        m = m*factor;
    }
    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        ncnn::Mat in;
        resize_bilinear(img, in, ws, hs);
        ncnn::Extractor ex = Pnet.create_extractor();
        //ex.set_num_threads(2);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        vector<FaceBox> boundingBox_;
        generateFaceBox(score_, location_, boundingBox_, scales_[i]);
        nms(boundingBox_, nms_threshold[0]);
        firstFaceBox_.insert(firstFaceBox_.end(), boundingBox_.begin(), boundingBox_.end());
        boundingBox_.clear();
    }
}

void MTCNN::RNet(){
    secondFaceBox_.clear();
    int count = 0;
    for(vector<FaceBox>::iterator it=firstFaceBox_.begin(); it!=firstFaceBox_.end();it++){
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 24, 24);
        ncnn::Extractor ex = Rnet.create_extractor();
		//ex.set_num_threads(2);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, FaceBox;
        ex.extract("prob1", score);
        ex.extract("conv5-2", FaceBox);
		if ((float)score[1] > threshold[1]) {
			for (int channel = 0; channel<4; channel++) {
				it->regreCoord[channel] = (float)FaceBox[channel];//*(FaceBox.data+channel*FaceBox.cstep);
			}
			it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
			it->score = score.channel(1)[0];//*(score.data+score.cstep);
			secondFaceBox_.push_back(*it);
		}
    }
}

float MTCNN::rnet(ncnn::Mat& img) {

	ncnn::Extractor ex = Rnet.create_extractor();
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
	img.substract_mean_normalize(mean_vals, norm_vals);
	ex.set_light_mode(true);
	ex.input("data", img);
	ncnn::Mat score;
	ex.extract("prob1", score);
	return (float)score[1];
}

void MTCNN::ONet(){
    thirdFaceBox_.clear();
    for(vector<FaceBox>::iterator it=secondFaceBox_.begin(); it!=secondFaceBox_.end();it++){
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 48, 48);
        ncnn::Extractor ex = Onet.create_extractor();
		//ex.set_num_threads(2);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, FaceBox, keyPoint;
        ex.extract("prob1", score);
        ex.extract("conv6-2", FaceBox);
        ex.extract("conv6-3", keyPoint);
		if ((float)score[1] > threshold[2]) {
			for (int channel = 0; channel < 4; channel++) {
				it->regreCoord[channel] = (float)FaceBox[channel];
			}
			it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
			it->score = score.channel(1)[0];
			for (int num = 0; num<5; num++) {
				(it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * keyPoint[num];
				(it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1) * keyPoint[num + 5];
			}
			thirdFaceBox_.push_back(*it);
		}
    }
}

FaceBox MTCNN::onet(ncnn::Mat& img, int x, int y, int w, int h) {

	FaceBox faceFaceBox;
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
	img.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Extractor ex = Onet.create_extractor();

	ex.set_light_mode(true);
	ex.input("data", img);
	ncnn::Mat score, FaceBox, keyPoint;
	ex.extract("prob1", score);
	ex.extract("conv6-2", FaceBox);
	ex.extract("conv6-3", keyPoint);
	faceFaceBox.score = score.channel(1)[0];
	faceFaceBox.x1 = static_cast<int>(FaceBox[0] * w) + x;
	faceFaceBox.y1 = static_cast<int>(FaceBox[1] * h) + y;
	faceFaceBox.x2 = static_cast<int>(FaceBox[2] * w) + h + x;
	faceFaceBox.y2 = static_cast<int>(FaceBox[3] * h) + h + y;
	for (int num = 0; num<5; num++) {
		(faceFaceBox.ppoint)[num] = x + w * keyPoint[num];
		(faceFaceBox.ppoint)[num + 5] = y + h * keyPoint[num + 5];
	}

	return faceFaceBox;
	
}

void MTCNN::detect(ncnn::Mat& img_, vector<FaceBox>& finalFaceBox_){
    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

    PNet();
    //the first stage's nms
    if(firstFaceBox_.size() < 1) return;
    nms(firstFaceBox_, nms_threshold[0]);
    refine(firstFaceBox_, img_h, img_w, true);
    //printf("firstFaceBox_.size()=%d\n", firstFaceBox_.size());


    //second stage
    RNet();
    //printf("secondFaceBox_.size()=%d\n", secondFaceBox_.size());
    if(secondFaceBox_.size() < 1) return;
    nms(secondFaceBox_, nms_threshold[1]);
    refine(secondFaceBox_, img_h, img_w, true);

    //third stage 
    ONet();
    //printf("thirdFaceBox_.size()=%d\n", thirdFaceBox_.size());
    if(thirdFaceBox_.size() < 1) return;
    refine(thirdFaceBox_, img_h, img_w, true);
    nms(thirdFaceBox_, nms_threshold[2], "Min");
    finalFaceBox_ = thirdFaceBox_;
	if (smooth)
		SmoothFaceBox(finalFaceBox_);
}

void MTCNN::detectMaxFace(ncnn::Mat& img_, vector<FaceBox>& finalFaceBox) {
	firstPreviousFaceBox_.clear();
	secondPreviousFaceBox_.clear();
	thirdPrevioussFaceBox_.clear();
	firstFaceBox_.clear();
	secondFaceBox_.clear();
	thirdFaceBox_.clear();

	//norm
	img = img_;
	img_w = img.w;
	img_h = img.h;
	img.substract_mean_normalize(mean_vals, norm_vals);

	//pyramid size
	float minl = img_w < img_h ? img_w : img_h;
	float m = (float)MIN_DET_SIZE / minsize;
	minl *= m;
	float factor = pre_facetor;
	vector<float> scales_;
	while (minl>MIN_DET_SIZE) {
		scales_.push_back(m);
		minl *= factor;
		m = m*factor;
	}
	sort(scales_.begin(), scales_.end());
	//printf("scales_.size()=%d\n", scales_.size());

	//Change the sampling process.
	for (size_t i = 0; i < scales_.size(); i++)
	{
		//first stage
		PNet(scales_[i]);
		nms(firstFaceBox_, nms_threshold[0]);
		nmsTwoBoxs(firstFaceBox_, firstPreviousFaceBox_, nms_threshold[0]);
		if (firstFaceBox_.size() < 1) {
			firstFaceBox_.clear();
			continue;
		}
		firstPreviousFaceBox_.insert(firstPreviousFaceBox_.end(), firstFaceBox_.begin(), firstFaceBox_.end());
		refine(firstFaceBox_, img_h, img_w, true);
		//printf("firstFaceBox_.size()=%d\n", firstFaceBox_.size());

		//second stage
		RNet();
		nms(secondFaceBox_, nms_threshold[1]);
		nmsTwoBoxs(secondFaceBox_, secondPreviousFaceBox_, nms_threshold[0]);
		secondPreviousFaceBox_.insert(secondPreviousFaceBox_.end(), secondFaceBox_.begin(), secondFaceBox_.end());
		if (secondFaceBox_.size() < 1) {
			firstFaceBox_.clear();
			secondFaceBox_.clear();
			continue;
		}
		refine(secondFaceBox_, img_h, img_w, true);

		//third stage
		ONet();

		if (thirdFaceBox_.size() < 1) {
			firstFaceBox_.clear();
			secondFaceBox_.clear();
			thirdFaceBox_.clear();
			continue;
		}
		refine(thirdFaceBox_, img_h, img_w, true);
		nms(thirdFaceBox_, nms_threshold[2], "Min");

		if (thirdFaceBox_.size() > 0) {
			extractMaxFace(thirdFaceBox_);
			finalFaceBox = thirdFaceBox_;//if largest face size is similar,.
			if (smooth)
				SmoothFaceBox(finalFaceBox);
			break;
		}
	}

}

float MTCNN::iou(FaceBox & b1, FaceBox & b2, string modelname)
{
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	maxX = max(b1.x1, b2.x1);
	maxY = max(b1.y1, b2.y1);
	minX = min(b1.x2, b2.x2);
	minY = min(b1.y2, b2.y2);
	//maxX1 and maxY1 reuse
	maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
	maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
	IOU = maxX * maxY;

	if (!modelname.compare("Union"))
		IOU = IOU / (b1.area + b2.area - IOU);
	else if (!modelname.compare("Min")) {
		IOU = IOU / ((b1.area < b2.area) ? b1.area : b2.area);
	}
	return IOU;
}

void MTCNN::SmoothFaceBox(std::vector<FaceBox>& finalFaceBox)
{
	static std::vector<FaceBox> preFaceBox_;
	for (int i = 0; i < finalFaceBox.size(); i++) {
		for (int j = 0; j < preFaceBox_.size(); j++) {
			if (iou(finalFaceBox[i], preFaceBox_[j]) > 0.90)
			{
				finalFaceBox[i] = preFaceBox_[j];
			}
			else if (iou(finalFaceBox[i], preFaceBox_[j]) > 0.6) {
				finalFaceBox[i].x1 = (finalFaceBox[i].x1 + preFaceBox_[j].x1) / 2;
				finalFaceBox[i].y1 = (finalFaceBox[i].y1 + preFaceBox_[j].y1) / 2;
				finalFaceBox[i].x2 = (finalFaceBox[i].x2 + preFaceBox_[j].x2) / 2;
				finalFaceBox[i].y2 = (finalFaceBox[i].y2 + preFaceBox_[j].y2) / 2;
				//finalFaceBox[i].area = (finalFaceBox[i].x2 - finalFaceBox[i].x1)*(finalFaceBox[i].y2 - finalFaceBox[i].y1);
				for (int k = 0; k < 10; k++)
				{
					finalFaceBox[i].ppoint[k] = (finalFaceBox[i].ppoint[k] + preFaceBox_[j].ppoint[k]) / 2;
				}
			}
		}
	}
	preFaceBox_ = finalFaceBox;

}
