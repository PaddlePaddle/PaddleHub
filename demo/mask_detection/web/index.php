<?php
$json_string = file_get_contents(".\\video\\result\\2-mask_detection.json");
// 用参数true把JSON字符串强制转成PHP数组 
$data_o = json_decode($json_string, true);
?>
<script language="javascript">
var pic_data = new Array();
<?php
for($i=0;$i<count($data_o);$i++){
	$frame=$data_o[$i]["frame"];
	$data_1=$data_o[$i]["data"];
	for($j=0;$j<count($data_1);$j++){
		?>
		 pic_data[<?php echo $frame;?>] = <?php echo json_encode($data_1);?>;
		// pic_data[<?php echo $frame;?>] = 1;
		<?php
	}
}
?>
</script>
<?php 
//摄像机位置，及视野宽高
$cams = array();

//雷达区图片
$area_w=250;
$area_h=500;

//cam图片大小
$cam_w=21;
$cam_h=23;

//vide宽高
$vide_w=1280;
$vide_h=720;

$cams[0]["cam_x"]=645;
$cams[0]["cam_y"]=243;
$cams[0]["cam_area_rotate"]=270;
$cams[0]["cam_area_w"]=0.4;	//比例
$cams[0]["cam_area_h"]=0.8;

$cams[1]["cam_x"]=888;
$cams[1]["cam_y"]=167;
$cams[1]["cam_area_rotate"]=180;
$cams[1]["cam_area_w"]=0.6;	//比例
$cams[1]["cam_area_h"]=1;

$cams[2]["cam_x"]=1051;
$cams[2]["cam_y"]=163;
$cams[2]["cam_area_rotate"]=270;
$cams[2]["cam_area_w"]=0.3;	//比例
$cams[2]["cam_area_h"]=1;

$cams[3]["cam_x"]=893;
$cams[3]["cam_y"]=257;
$cams[3]["cam_area_rotate"]=275;
$cams[3]["cam_area_w"]=1;	//比例
$cams[3]["cam_area_h"]=1;

$cams[4]["cam_x"]=714;
$cams[4]["cam_y"]=304;
$cams[4]["cam_area_rotate"]=90;
$cams[4]["cam_area_w"]=0.2;	//比例
$cams[4]["cam_area_h"]=0.5;

?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8" />
<meta http-equiv="X-UA-Compatible" content="IE=Edge,chrome=1" />
<meta name="viewport" content="width=device-width" />
<meta name="format-detection" content="telephone=no, address=no, email=no" />
<link rel="stylesheet" type="text/css" href="css/css.css">
<script type="text/javascript" src="js/jquery.js"></script>
<script type="text/javascript" src="js/echarts.js"></script>
<style type="text/css">
body {
	margin: 0;
	padding: 0px;
	font-family: "Microsoft YaHei", YaHei, "微软雅黑", SimHei, "黑体";
	font-size: 14px;
}
.box {

}
.cam {
	position: absolute;
	border: none;
}
.area {
	cursor: pointer;
	z-index: 999;
}
</style>
</head>
<body>
<div class="header">
	<div class="right-bar">
		<input type="text" class="search">
		<span class="icon icon1"></span> <span class="icon icon2"></span>
		<div class="avatar"><img src="image/ava.png" alt=""></div>
		<span class="arrow"></span> </div>
	<div class="logo"><img src="image/logo.png" alt="" style="font-size:120px;"></div>
</div>
<div class="bd">
	<div class="side">
		<div class="nav">
			<ul>
				<li><a href="#" class="nav1 current"></a></li>
				<li><a href="#" class="nav2"></a></li>
				<li><a href="#" class="nav3"></a></li>
			</ul>
		</div>
	</div>
	
	<div class="con">
		<div class="hd clear">
			<div class="right-bar">摄像头分组显示 <span class="select-wrap">
				<select name="">
					<option value="A区摄像头">A区摄像头</option>
					<option value="A区摄像头">A区摄像头</option>
				</select>
				</span> </div>
			<span class="tit">摄像头_1</span> </div>
		<div class="con-bd">
			<div class="map-wrap">
				<div class="zoom-bar"> <span class="icon zoom "></span> <span class="icon mouse"></span> <span class="icon narrow"></span> </div>
				<div class="box">
					<table width="1870" border="0" cellspacing="10">
						<tr>
							<td width="44%" valign="top"><p></p>
								
								<!-- （拖拽在回来可能会有问题 TODO） -->
								
								<div id="video" style="width: 1000px; height:560px;"></div>
								<script type="text/javascript" src="js/ckplayer/ckplayer.js" charset="UTF-8"></script> 
								<script type="text/javascript">
									
									var videoObject = {
										container: '#video', //容器的ID或className
										variable: 'player', //播放函数名称
										loop: true, //播放结束是否循环播放
										loaded: "loadedHandler",
										autoplay: true,//是否自动播放					
										drag: 'start', //拖动的属性
										video: [
											['video/result/1-mask_detection.mp4', 'video/mp4', '中文标清', 0],
										]
									};
									var player = new ckplayer(videoObject);
									function timeHandler(time) {
										$(".no_pmask").html("");
										$(".pmask").html("");
										
										var rtime = parseInt(time*1000);
										frame=parseInt(time*30); //每秒30帧 第多少帧
										for (i = 0; i < pic_data[frame].length; i++) { 
											if(pic_data[frame][i].label=="NO MASK"){
												$(".no_pmask").append("<img src='video/result/"+pic_data[frame][i].img+"' height=100>");
											} else {
												$(".pmask").append("<img src='video/result/"+pic_data[frame][i].img+"' height=100>");
											}
										}
									}
									function seekTimeHandler(time) {
										var rtime = parseInt(time*1000);
										$(".seekstate").html(rtime);
									}
									
									function loadedHandler() {
										player.addListener('time', timeHandler); //监听播放时间
										player.addListener('seekTime', seekTimeHandler); //监听跳转播放完
				
									/*
										player.addListener('error', errorHandler); //监听视频加载出错
										player.addListener('loadedmetadata', loadedMetaDataHandler); //监听元数据
										player.addListener('duration', durationHandler); //监听播放时间
										player.addListener('play', playHandler); //监听暂停播放
										player.addListener('pause', pauseHandler); //监听暂停播放
										player.addListener('buffer', bufferHandler); //监听缓冲状态
										player.addListener('seek', seekHandler); //监听跳转播放完成
										成
										player.addListener('volume', volumeChangeHandler); //监听音量改变
										player.addListener('full', fullHandler); //监听全屏/非全屏切换
										player.addListener('ended', endedHandler); //监听播放结束
										player.addListener('screenshot', screenshotHandler); //监听截图功能
										player.addListener('mouse', mouseHandler); //监听鼠标坐标
										player.addListener('frontAd', frontAdHandler); //监听前置广告的动作
										player.addListener('wheel', wheelHandler); //监听视频放大缩小
										player.addListener('controlBar', controlBarHandler); //监听控制栏显示隐藏事件
										player.addListener('clickEvent', clickEventHandler); //监听点击事件
										player.addListener('definitionChange', definitionChangeHandler); //监听清晰度切换事件
										player.addListener('speed', speedHandler); //监听加载速度*/
									}
								</script>
								<br>
								</td>
							<td width="56%" valign="top">
							
							<p style="padding:0 0 0 40px; font-size:30px; color:red; backgroud-color:#b3b6c7">无口罩</p>
							<p class="no_pmask" style="padding:40px;"></p><br><br><br><br>
							<p style="padding:0 0 0 40px; font-size:30px; color:#08d814; backgroud-color:#b3b6c7">有口罩</p>
							<p class="pmask" style="padding:40px; ;"></p>
							</td>
							</tr>
							<tr>
								<td colspan="2" valign="top"><div style="position:relative;"><img src="image/b2f2.png" width="1250"/> 
									<?php for($i=0;$i<count($cams);$i++){
										$r =  get_mast_postion($cams[$i]["cam_x"],$cams[$i]["cam_y"],$cams[$i]["cam_area_w"],$cams[$i]["cam_area_h"],$area_w,$area_h,$cam_w,$cam_h);
									?>
									<div>
										<div style="border: 2px #fff solid; border: none;  position: absolute; left: <?php echo $r['left'];?>px; top: <?php echo $r['top'];?>px;  transform: rotate(<?php echo $cams[$i]["cam_area_rotate"];?>deg);"> <img src="image/mask.png" width="<?php echo $area_w*$cams[$i]["cam_area_w"];?>" height="<?php echo $area_h*$cams[$i]["cam_area_h"];?>" />
											
										</div>
										<div id="cam<?php echo $i?>" class="cam area" style="left: <?php echo $cams[$i]["cam_x"];?>px;top: <?php echo $cams[$i]["cam_y"];?>px;"> <img src="image/cam.png" width="<?php echo $cam_w;?>" height="<?php echo $cam_h;?>"/> </div>
									</div>
									<?php }?>
								</div></td>
						</tr>
					</table>
				</div>
			</div>
		</div>
	</div>
</div>
</body>
</html>

<?php 
function get_mast_postion($cam_x,$cam_y,$cam_area_w,$cam_area_h,$area_w,$area_h,$cam_w,$cam_h){
	$left=$cam_x-$cam_area_w*$area_w/2;
	$top=$cam_y-$cam_area_h*$area_h/2;
	
	$r['left'] = $left+$cam_w/2;
	$r['top'] = $top+$cam_h/2;

	return $r;
}
?>
